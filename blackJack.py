import cv2
import inference
import supervision as sv
import openai

# Replace with your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to read goals/instructions from a text file
def read_goals_from_file(file_path):
    with open(file_path, 'r') as file:
        goals = file.read()
    return goals

# Create an OpenAI GPT-4 conversation with the goals
def create_gpt4_conversation(goals):
    conversation = [
        {"role": "system", "content": "You are a robot that can interact with the webcam."},
        {"role": "user", "content": goals},
    ]
    return conversation

# Function to handle OpenAI GPT-4 responses
def handle_gpt4_response(response):
    # Extract and print the GPT-4 model's response
    model_response = response['choices'][0]['message']['content']
    print("GPT-4 Response:", model_response)

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    detections = detections[detections.confidence > 0.45]
    print(detections)
    
    # Read goals from the text file
    goals = read_goals_from_file("goals.txt")
    
    # Create a GPT-4-turbo conversation with the goals
    conversation = create_gpt4_conversation(goals)
    
    # Use GPT-4-turbo to get a response
    gpt4_response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=conversation
    )
    
    # Handle the GPT-3 response
    handle_gpt4_response(gpt4_response)
    
    # Display the annotated image
    cv2.imshow(
        "Prediction", 
        annotator.annotate(
            scene=image, 
            detections=detections,
            labels=labels
        )
    )
    cv2.waitKey(1)

inference.Stream(
    source="webcam",  # or rtsp stream or camera id
    model="black-jack-8goxw/9",  # from Universe
    output_channel_order="BGR",
    use_main_thread=True,  # for OpenCV display
    on_prediction=on_prediction,
)