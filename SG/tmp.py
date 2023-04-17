import PySimpleGUI as sg

# Define the options for the databases
options = {
    "emodb": "EMODB",
    "ravdess": "RAVDESS",
    "savee": "SAVEE"
}

# Define the options for the emotions
emotions = {
    "angry": "Angry",
    "happy": "Happy",
    "neutral": "Neutral",
    "pleasantSurprise": "Pleasant Surprise",
    "sad": "Sad"
}

# Define the layout of the dialog
layout = [
    [sg.Text("Select the training database")],
    [sg.Combo(list(options.values()), key="train_db")],
    [sg.Text("Select the testing database")],
    [sg.Combo(list(options.values()), key="test_db")],
    [sg.Text("Select the emotions to use for training")],
    [sg.Column([[sg.Checkbox(emotions[e], key=f"train_{e}")] for e in emotions])],
    [sg.Text("Select the emotions to use for testing")],
    [sg.Column([[sg.Checkbox(emotions[e], key=f"test_{e}")] for e in emotions])],
    [sg.Button("OK"), sg.Button("Cancel")]
]

# Create the dialog
window = sg.Window("Select Databases and Emotions", layout)

# Initialize the selected databases and emotions lists
selected_databases = []
selected_emotions = []

# Loop until the user clicks the OK button or closes the dialog
while True:
    event, values = window.read()

    if event in (None, "Cancel"):
        break

    # Check if the user has selected the same database twice
    if values["train_db"] == values["test_db"]:
        sg.popup("Please select different databases for training and testing.")
        continue

    # Check if the user has already selected the same database
    if values["train_db"] in selected_databases or values["test_db"] in selected_databases:
        sg.popup("Please select different databases.")
        continue

    # Clear the selected emotions list
    selected_emotions.clear()

    # Add the selected emotions to the list
    for e in emotions:
        if values[f"train_{e}"]:
            selected_emotions.append(e)
        if values[f"test_{e}"]:
            selected_emotions.append(e)

    # Check if the user has selected at least one emotion
    if len(selected_emotions) == 0:
        sg.popup("Please select at least one emotion.")
        continue

    # Add the selected databases to the list
    selected_databases.append(values["train_db"])
    selected_databases.append(values["test_db"])

    # Remove the selected databases from the options
    options.pop(values["train_db"])
    options.pop(values["test_db"])

    # If the user has selected two databases, exit the loop
    if len(selected_databases) == 2:
        break

# Close the dialog
window.close()

# Print the selected databases and emotions
print("Training database:", selected_databases[0])
print("Testing database:", selected_databases[1])
print("Selected emotions:", selected_emotions)
##
import PySimpleGUI as sg

# Define the options for the emotions
emotions = ["angry", "happy", "neutral", "pleasantSuprise", "sad"]

# Define the layout of the dialog
layout = [
    [sg.Text("Select the emotions to use for training")],
    [sg.Column([[sg.Checkbox(e.capitalize(), key=f"train_{e}")] for e in emotions])],
    [sg.Text("Select the emotions to use for testing")],
    [sg.Column([[sg.Checkbox(e.capitalize(), key=f"test_{e}")] for e in emotions])],
    [sg.Button("OK"), sg.Button("Cancel")]
]

# Create the dialog
window = sg.Window("Select Emotions", layout)

# Loop until the user clicks the OK button or closes the dialog
while True:
    event, values = window.read()

    if event in (None, "Cancel"):
        break

    # Clear the selected emotions list
    selected_emotions = []

    # Add the selected emotions to the list
    for e in emotions:
        if values[f"train_{e}"]:
            selected_emotions.append(e)
        if values[f"test_{e}"]:
            selected_emotions.append(e)

    # Check if the user has selected at least one emotion
    if len(selected_emotions) == 0:
        sg.popup("Please select at least one emotion.")
        continue

    # If the user has selected at least one emotion, exit the loop
    break

# Close the dialog
window.close()

# Print the selected emotions
print("Selected emotions:", selected_emotions)
