from flask import Flask, request, jsonify, send_file, redirect
import cv2
import numpy as np
import os
import uuid
import tempfile
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

# Create a temp directory for the duration of the instance
TEMP_DIR = tempfile.mkdtemp()
UPLOAD_FOLDER = os.path.join(TEMP_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(TEMP_DIR, 'processed')

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper Functions for Fog Removal
def estimate_atmospheric_light(img, mean_of_top_percentile=0.1):
    # Find the number of pixels to take from the top percentile
    num_pixels = int(np.prod(img.shape[:2]) * mean_of_top_percentile)

    # Find the maximum pixel value for each channel
    max_channel_vals = np.max(np.max(img, axis=0), axis=0)

    # Sort the channel values in descending order
    sorted_vals = np.argsort(max_channel_vals)[::-1]

    # Take the highest pixel values from each channel
    atmospheric_light = np.zeros((1, 1, 3), np.uint8)
    for channel in range(3):
        atmospheric_light[0, 0, channel] = np.sort(img[:, :, sorted_vals[channel]].ravel())[-num_pixels]

    return atmospheric_light

def guided_filter(I, p, radius, eps):
    """
    Simple implementation of guided filter
    I: guidance image (3-channel)
    p: filtering input (1-channel)
    radius: window radius
    eps: regularization parameter
    """
    mean_I_r = cv2.boxFilter(I[:,:,0], cv2.CV_64F, (radius, radius), normalize=True)
    mean_I_g = cv2.boxFilter(I[:,:,1], cv2.CV_64F, (radius, radius), normalize=True)
    mean_I_b = cv2.boxFilter(I[:,:,2], cv2.CV_64F, (radius, radius), normalize=True)
    
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius), normalize=True)
    
    mean_Ip_r = cv2.boxFilter(I[:,:,0]*p, cv2.CV_64F, (radius, radius), normalize=True)
    mean_Ip_g = cv2.boxFilter(I[:,:,1]*p, cv2.CV_64F, (radius, radius), normalize=True)
    mean_Ip_b = cv2.boxFilter(I[:,:,2]*p, cv2.CV_64F, (radius, radius), normalize=True)
    
    # covariance
    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    
    # variance
    var_I_rr = cv2.boxFilter(I[:,:,0]*I[:,:,0], cv2.CV_64F, (radius, radius), normalize=True) - mean_I_r * mean_I_r
    var_I_rg = cv2.boxFilter(I[:,:,0]*I[:,:,1], cv2.CV_64F, (radius, radius), normalize=True) - mean_I_r * mean_I_g
    var_I_rb = cv2.boxFilter(I[:,:,0]*I[:,:,2], cv2.CV_64F, (radius, radius), normalize=True) - mean_I_r * mean_I_b
    var_I_gg = cv2.boxFilter(I[:,:,1]*I[:,:,1], cv2.CV_64F, (radius, radius), normalize=True) - mean_I_g * mean_I_g
    var_I_gb = cv2.boxFilter(I[:,:,1]*I[:,:,2], cv2.CV_64F, (radius, radius), normalize=True) - mean_I_g * mean_I_b
    var_I_bb = cv2.boxFilter(I[:,:,2]*I[:,:,2], cv2.CV_64F, (radius, radius), normalize=True) - mean_I_b * mean_I_b
    
    a = np.zeros((p.shape[0], p.shape[1], 3))
    
    # Using matrix operations instead of loops for better performance
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            Sigma = np.array([
                [var_I_rr[i,j] + eps, var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j] + eps, var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j] + eps]
            ])
            cov = np.array([cov_Ip_r[i,j], cov_Ip_g[i,j], cov_Ip_b[i,j]])
            a[i,j] = np.linalg.solve(Sigma, cov)
    
    b = mean_p - a[:,:,0] * mean_I_r - a[:,:,1] * mean_I_g - a[:,:,2] * mean_I_b
    
    # filter
    mean_a0 = cv2.boxFilter(a[:,:,0], cv2.CV_64F, (radius, radius), normalize=True)
    mean_a1 = cv2.boxFilter(a[:,:,1], cv2.CV_64F, (radius, radius), normalize=True)
    mean_a2 = cv2.boxFilter(a[:,:,2], cv2.CV_64F, (radius, radius), normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius), normalize=True)
    
    q = mean_a0 * I[:,:,0] + mean_a1 * I[:,:,1] + mean_a2 * I[:,:,2] + mean_b
    
    return q

def fast_visibility_restoration(frame, atmospheric_light, tmin=0.1, A=1.0, omega=0.95, guided_filter_radius=40, gamma=0.7):
    # Normalize the frame and atmospheric light
    normalized_frame = frame.astype(np.float32) / 255.0
    normalized_atmospheric_light = atmospheric_light.astype(np.float32) / 255.0

    # Compute the transmission map
    transmission_map = 1 - omega * cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2GRAY) / cv2.cvtColor(normalized_atmospheric_light, cv2.COLOR_BGR2GRAY)

    # Apply our custom guided filter to the transmission map
    transmission_map = guided_filter(normalized_frame, transmission_map, guided_filter_radius, eps=1.0)

    # Apply the gamma correction to the transmission map
    transmission_map = np.power(transmission_map, gamma)

    # Threshold the transmission map to ensure a minimum value
    transmission_map = np.maximum(transmission_map, tmin)

    # Add an extra dimension to transmission_map for broadcasting
    transmission_map_3d = np.expand_dims(transmission_map, axis=2)

    # Compute the dehazed image
    dehazed_frame = (normalized_frame - normalized_atmospheric_light) / transmission_map_3d + normalized_atmospheric_light

    # Apply the A parameter to the dehazed image
    dehazed_frame = A * dehazed_frame

    # Normalize the dehazed image and convert to 8-bit color
    dehazed_frame = np.uint8(np.clip(dehazed_frame * 255.0, 0, 255))

    return dehazed_frame

@app.route('/process_video', methods=['POST'])
def process_video():
    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({'error': 'No video file in the request'}), 400
    
    file = request.files['video']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No video selected for processing'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid collisions
        unique_id = str(uuid.uuid4())
        input_filename = secure_filename(file.filename)
        base_filename, extension = os.path.splitext(input_filename)
        
        # Set file paths
        input_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}_{unique_id}{extension}")
        output_path = os.path.join(PROCESSED_FOLDER, f"{base_filename}_dehazed_{unique_id}{extension}")
        
        # Save the uploaded file
        file.save(input_path)
        
        try:
            # Process the video
            # Read the video file
            video_capture = cv2.VideoCapture(input_path)
            
            # Get video properties
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read the first frame and estimate atmospheric light
            ret, frame = video_capture.read()
            if not ret:
                return jsonify({'error': 'Could not read the video file'}), 400
                
            atmospheric_light = estimate_atmospheric_light(frame)
            
            # Create a video writer object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Reset the video capture to the beginning
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process each frame
            frame_count = 0
            max_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process up to 1800 frames (about 1 minute at 30fps) to avoid timeout
            max_frames = min(max_frames, 1800)
            
            while frame_count < max_frames:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Apply the dehazing algorithm
                dehazed_frame = fast_visibility_restoration(frame, atmospheric_light)
                
                # Write the processed frame to the output video
                out.write(dehazed_frame)
                frame_count += 1
            
            # Release resources
            video_capture.release()
            out.release()
            
            # Generate a video URL that points to our server
            video_url = f"/get_processed_video/{os.path.basename(output_path)}"
            
            # If this was a request from the web form, redirect to the result
            if request.headers.get('Accept', '').find('text/html') != -1:
                return redirect(f'/result?video_url={video_url}')
            
            # Otherwise return JSON for API clients
            return jsonify({
                'success': True, 
                'message': 'Video processed successfully',
                'video_url': video_url
            })
            
        except Exception as e:
            # Clean up the input file in case of error
            try:
                os.remove(input_path)
            except:
                pass
                
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/get_processed_video/<filename>', methods=['GET'])
def get_processed_video(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), 
                    mimetype='video/mp4', 
                    as_attachment=True, 
                    download_name=filename)

@app.route('/result', methods=['GET'])
def result():
    video_url = request.args.get('video_url', '')
    return f'''
    <html>
        <head>
            <title>Fog Removal Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333366; }}
                .video-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Fog Removal Result</h1>
            <div class="video-container">
                <video width="640" height="480" controls>
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            <p>
                <a href="/">Process another video</a> | 
                <a href="{video_url}" download>Download processed video</a>
            </p>
        </body>
    </html>
    '''

@app.route('/', methods=['GET'])
def home():
    return '''
    <html>
        <head>
            <title>Fog Removal API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333366; }
                form { margin: 20px 0; }
                .info { margin: 20px 0; color: #666; }
                .warning { color: #cc3300; }
            </style>
        </head>
        <body>
            <h1>Fog Removal API</h1>
            <form action="/process_video" method="post" enctype="multipart/form-data">
                <div>
                    <label for="video">Select a foggy video:</label>
                    <input type="file" id="video" name="video" accept="video/*">
                </div>
                <div style="margin-top: 10px;">
                    <button type="submit">Process Video</button>
                </div>
            </form>
            <div class="info">
                <p>This API removes fog from videos using a visibility restoration algorithm.</p>
                <p class="warning">Note: Processing may take several minutes depending on video length and resolution.</p>
                <p class="warning">For best results, use videos under 1 minute in length.</p>
            </div>
        </body>
    </html>
    '''
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'API is working!'})

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=PORT, debug=False)
