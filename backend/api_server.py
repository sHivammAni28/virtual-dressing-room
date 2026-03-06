from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import threading
import os
import sys
import signal
import time
import logging
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from email_service import email_service
from validation_utils import ContactFormValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

current_process = None
process_lock = threading.Lock()


def map_shirt_id_to_selection(shirt_id):
    if 1 <= shirt_id <= 7:
        return "male", shirt_id
    if 101 <= shirt_id <= 105:
        return "female", shirt_id - 100
    raise ValueError(f"Invalid shirt ID: {shirt_id}")


def test_camera_access():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Camera could not be opened"
        ret, _ = cap.read()
        cap.release()
        if not ret:
            return False, "Could not read frame from camera"
        return True, "Camera is accessible"
    except Exception as exc:
        return False, f"Camera test failed: {exc}"


def start_tryon_service(gender, shirt_index):
    global current_process

    try:
        stop_current_process()

        camera_ok, camera_msg = test_camera_access()
        if not camera_ok:
            logger.error(f"Camera test failed: {camera_msg}")
            return False, camera_msg

        logger.info(f"Camera test passed: {camera_msg}")

        script_path = os.path.join(os.path.dirname(__file__), 'tryon_service.py')

        with process_lock:
            current_process = subprocess.Popen(
                [
                    sys.executable,
                    script_path,
                    '--gender', gender,
                    '--shirt-index', str(shirt_index)
                ],
                cwd=os.path.dirname(__file__),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

        logger.info(f"Started try-on service for {gender} shirt {shirt_index} (PID: {current_process.pid})")

        monitor_thread = threading.Thread(target=monitor_process_output, daemon=True)
        monitor_thread.start()

        return True, "Try-on service started successfully"

    except Exception as exc:
        logger.error(f"Failed to start try-on service: {exc}")
        return False, f"Failed to start try-on service: {exc}"


def monitor_process_output():
    global current_process

    if current_process is None:
        return

    try:
        while current_process.poll() is None:
            if current_process.stdout:
                line = current_process.stdout.readline()
                if line:
                    logger.info(f"TryOn Process: {line.strip()}")

            if current_process.stderr:
                line = current_process.stderr.readline()
                if line:
                    logger.error(f"TryOn Process Error: {line.strip()}")

            time.sleep(0.1)
    except Exception as exc:
        logger.error(f"Error monitoring process output: {exc}")


def stop_current_process():
    global current_process

    with process_lock:
        if current_process and current_process.poll() is None:
            try:
                logger.info(f"Stopping try-on process (PID: {current_process.pid})")
                current_process.terminate()
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Graceful termination failed, force killing process")
                current_process.kill()
                current_process.wait()
            except Exception as exc:
                logger.error(f"Error stopping process: {exc}")
            finally:
                current_process = None
                logger.info("Try-on process stopped")


def try_on():
    try:
        data = request.get_json()
        shirt_id = data.get('shirtId') if data else None

        logger.info(f"Received try-on request for shirt ID: {shirt_id}")

        if not shirt_id:
            return jsonify({'success': False, 'message': 'Shirt ID is required'}), 400

        try:
            gender, shirt_index = map_shirt_id_to_selection(shirt_id)
            logger.info(f"Mapped shirt ID {shirt_id} to {gender} shirt {shirt_index}")
        except ValueError as exc:
            logger.error(f"Invalid shirt ID mapping: {exc}")
            return jsonify({'success': False, 'message': str(exc)}), 400

        success, message = start_tryon_service(gender, shirt_index)

        if success:
            return jsonify({'success': True, 'message': message, 'gender': gender, 'shirtIndex': shirt_index})
        return jsonify({'success': False, 'message': message}), 500

    except Exception as exc:
        logger.error(f"API error: {exc}")
        return jsonify({'success': False, 'message': f'Server error: {exc}'}), 500


def stop_tryon():
    try:
        logger.info("Received stop request")
        stop_current_process()
        return jsonify({'success': True, 'message': 'Virtual try-on stopped'})
    except Exception as exc:
        logger.error(f"Error stopping try-on: {exc}")
        return jsonify({'success': False, 'message': f'Error stopping try-on: {exc}'}), 500


def get_status():
    global current_process

    with process_lock:
        is_running = current_process is not None and current_process.poll() is None
        pid = current_process.pid if current_process else None

    return jsonify({'isRunning': is_running, 'pid': pid, 'timestamp': datetime.now().isoformat()})


def test_camera():
    try:
        camera_ok, camera_msg = test_camera_access()
        return jsonify({'success': camera_ok, 'message': camera_msg})
    except Exception as exc:
        return jsonify({'success': False, 'message': f'Camera test error: {exc}'}), 500


def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Virtual Try-On API Server is running',
        'timestamp': datetime.now().isoformat()
    })


def contact_form():
    try:
        form_data = request.get_json()

        if not form_data:
            return jsonify({'success': False, 'message': 'No form data provided'}), 400

        logger.info(f"Received contact form submission from {form_data.get('companyName', 'Unknown')}")

        is_valid, errors, validated_data = ContactFormValidator.validate_contact_form(form_data)

        if not is_valid:
            logger.warning(f"Contact form validation failed: {errors}")
            return jsonify({'success': False, 'message': 'Form validation failed', 'errors': errors}), 400

        email_sent, email_message = email_service.send_contact_email(validated_data)
        if not email_sent:
            logger.warning(f"Email sending failed: {email_message}")

        confirmation_sent, confirmation_message = email_service.send_confirmation_email(validated_data)
        if not confirmation_sent:
            logger.warning(f"Failed to send confirmation email: {confirmation_message}")

        logger.info(f"Contact form processed successfully for {validated_data.get('companyName')}")

        if email_sent:
            message = 'Thank you for your inquiry! We will get back to you within 24 hours.'
        else:
            message = 'Thank you for your inquiry! Your form has been submitted successfully. (Email notification temporarily unavailable)'

        return jsonify({
            'success': True,
            'message': message,
            'emailSent': email_sent,
            'confirmationSent': confirmation_sent
        })

    except Exception as exc:
        logger.error(f"Contact form API error: {exc}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred. Please try again later.'}), 500


def get_contact_options():
    try:
        return jsonify({
            'success': True,
            'options': {
                'companySizes': ContactFormValidator.COMPANY_SIZES,
                'inquiryTypes': ContactFormValidator.INQUIRY_TYPES,
                'meetingModes': ContactFormValidator.MEETING_MODES,
                'countries': ContactFormValidator.COUNTRIES
            }
        })
    except Exception as exc:
        logger.error(f"Error getting contact options: {exc}")
        return jsonify({'success': False, 'message': 'Failed to load form options'}), 500


def cleanup():
    logger.info("Cleaning up...")
    stop_current_process()


def signal_handler(sig, frame):
    logger.info("Received shutdown signal")
    cleanup()
    sys.exit(0)


app.add_url_rule('/api/try-on', 'try_on', try_on, methods=['POST'])
app.add_url_rule('/api/stop', 'stop_tryon', stop_tryon, methods=['POST'])
app.add_url_rule('/api/status', 'get_status', get_status, methods=['GET'])
app.add_url_rule('/api/test-camera', 'test_camera', test_camera, methods=['GET'])
app.add_url_rule('/health', 'health_check', health_check, methods=['GET'])
app.add_url_rule('/api/contact', 'contact_form', contact_form, methods=['POST'])
app.add_url_rule('/api/contact/options', 'get_contact_options', get_contact_options, methods=['GET'])

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    logger.info("Starting Virtual Try-On API Server...")
    logger.info("Server will be available at: http://localhost:5000")
    logger.info("API endpoints:")
    logger.info("  POST /api/try-on - Start virtual try-on")
    logger.info("  POST /api/stop - Stop virtual try-on")
    logger.info("  GET /api/status - Check service status")
    logger.info("  GET /api/test-camera - Test camera access")
    logger.info("  POST /api/contact - Submit contact form")
    logger.info("  GET /api/contact/options - Get form dropdown options")
    logger.info("  GET /health - Health check")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cleanup()
