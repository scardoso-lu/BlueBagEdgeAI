import redis
import tkinter as tk
import threading
import queue

# -------------------------------
# Redis setup
# -------------------------------
# Host, port, and channel for Redis connection
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_CHANNEL = "yolo:light_state"

# Connect to Redis server
# decode_responses=True ensures strings are returned instead of bytes
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Create a pub/sub object to subscribe to a channel
pubsub = r.pubsub()
pubsub.subscribe(REDIS_CHANNEL)

# -------------------------------
# Thread-safe queue
# -------------------------------
# This queue will safely transfer messages from the Redis listener thread to the Tkinter main thread
redis_queue = queue.Queue()

# -------------------------------
# Redis listener thread
# -------------------------------
def redis_listener():
    """
    Continuously listens to the Redis channel.
    When a message arrives, put it in the queue for the main thread to process.
    """
    for message in pubsub.listen():
        if message["type"] == "message":
            # Normalize data: strip whitespace and convert to uppercase
            redis_queue.put(message["data"].strip().upper())

# Start the listener thread as daemon so it won't block program exit
threading.Thread(target=redis_listener, daemon=True).start()

# -------------------------------
# Tkinter window (MAIN THREAD)
# -------------------------------
# Set up the main GUI window
root = tk.Tk()
root.title("Garbage Classification Status")
root.geometry("400x400")

# Label to display the current state
label = tk.Label(
    root,
    text="OFF",
    font=("Arial", 48),
    fg="white",
    bg="gray"
)
label.pack(expand=True, fill="both")

# -------------------------------
# UI update logic
# -------------------------------
def update_ui(state):
    """
    Change the label text and background color based on the state.
    """
    if state == "TRUE":
        label.config(text="TRUE", bg="green")
    elif state == "FALSE":
        label.config(text="FALSE", bg="red")
    else:
        # Default state
        label.config(text="OFF", bg="gray")

# -------------------------------
# Poll Redis queue (non-blocking)
# -------------------------------
# Variables to track repeated states
prev_state = None
same_state = 0

def process_messages():
    """
    Check the Redis queue for new messages and update the UI accordingly.
    Ensures the UI only updates after receiving the same state multiple times
    to prevent flickering from transient values.
    """
    global prev_state, same_state

    try:
        while True:
            # Non-blocking read from queue
            state = redis_queue.get_nowait()
            # Remove periods that might come in messages
            state = state.replace('.', '')
            print(f"state: {state}")  # Debug output

            if state == "OFF":
                # Always immediately show OFF
                update_ui(state)
                same_state = 0
            elif state == prev_state:
                # Increment count if the state is repeated
                same_state += 1
            else:
                # Reset counter when a new state appears
                same_state = 0

            # Only update UI if the same state has been stable for 5 consecutive messages
            if same_state >= 5:
                update_ui(state)
                same_state = 0

            prev_state = state

    except queue.Empty:
        # No more messages to process
        pass

    # Schedule the next poll in 50ms
    root.after(50, process_messages)

# Start polling the queue
process_messages()

# Start the Tkinter main loop
root.mainloop()
