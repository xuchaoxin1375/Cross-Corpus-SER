welcome_message =  "Welcome to My App!"
choose_emotion_config =  "Please select an emotional combination for testing: recommended combinations are AS, HNS, AHNS, AHNPS. \nNote that there is a difference between 'surprise' and 'pleasantSurprise' in the SAVEE dataset, \nso the AHNPS combination is not recommended for use on SAVEE."
choose_feature_config =  "Please choose one or more features"
feature_transform_config =  "feature transformer config"
choose_algorithm =  "Choose an algorithm for testing"
choose_audio =  "Please select an audio sample file to recognize its emotion."
recognize_the_audio_emotion =  "Recognize the emotion of the selected audio file."
draw_diagram =  "choose one diagram to draw"
draw_diagram_detail =  "Draw the [waveform|spectrogram|Mel spectrogram] of the selected file:"
welcome_title =  "𝒲ℯ𝓁𝒸ℴ𝓂ℯ 𝓉ℴ ℯ𝓍𝓅ℯ𝓇𝒾ℯ𝓃𝒸ℯ 𝒞𝒞𝒮ℰℛ 𝒞𝓁𝒾ℯ𝓃𝓉!"
result_training =  "result of model training:"
train_result_title =  "Train Result"
result_frame_prompt =  "Emotion Of Select File(Predict Result)"
recognize_it =  "Recognize it"
WaveForm = "WaveForm"
FreqGraph = "FreqGraph"
MelFreqGraph = "MelFreqGraph"
algorithmes_chooser_title="Algorithms chooser"
choose_emotion_config_title="choose the emotion config"
emotion_config_legend="Emotion Config chooser"
feature_transform_legend="Feature Transform chooser"
feature_config_legend="Feature Config chooser"

n_components_tooltip="input the number of components to keep."
n_components_prompt='n_components:'
demension_prompt="feature_dimension:"
pending_prompt="pending"

other_parameter_legend="Other Parameter Settings"
select_training_db="Select the training database"
select_testing_db="Select the testing database"
files_selected_prompt="select multiple files,which will be shown here "
confirm_files_selected_button="confirm files selected"
recursively_scan_subdir="Recursively scan subdirectories"
auto_refresh="auto refresh"
auto_refresh_tooltip="click to manual refresh the files listbox"
short_path="short path"
filter_by_regex_prompt="Filter by regex:"
selected_audios_prompt="Selected audio files:"
no_files=f"0 files"
audios_filter="Audio File Filter"
filterd_audios="Filtered audio files: "
files_count=" files"
train_model_warning="please train the SER model and then try angin!"

selected_files_tooltip = """
you can observe the files your choosed in last listBox
Whether it is a continuous selection or a skip selection, 
these selected files will be tightly arranged and 
the number of files will be displayed at the top
"""
filter_tooltip = """
    the listbox of files allow you to choose one or more files \n using left button of your mouse, 
you can use `Ctrl+Click` to select multiple files(jump to the selected file is allowed too!)

    you can right click after you choose one or more files to do something like these: 
    1.file size
    2.file path(absolute path)
    3.recognize emotion
    4.play file(audio) you choosed
    *.all of above could work in multiple files one by one automatically
"""
not_exist="not exist!"
confirm_folder_selected= "confirm folder selected"
folder_browse="folder browse"
choose_folder_tooltip="choose a folder you want to do SER,\nthe default folder is "
files_browse="browse files"
current_directory_prompt="current directory:"
show_file_path="Show file path"
show_file_size="Show file size"
show_audio_duration="Show audio duration"
play_audio="Play Audio"
emotion_recognize="Emotion Recognize"
path_input_tooltip="you can paste or type a dir path!\n or use the right side Browse button to choose a dir"
click_filter_prompt="click filter or input regex to scan audio file!"
listbox_default_value_prompt = "hover your mouse in this listbox area to check tooltips!"
filter_audios="filter audios"
select_dir_prompt="Select a directory:"

save_patient_warning="click to save to a csv file!\nthe save operation will comsume  some time to complete!Be patient!"
close_table_prompt="if you want to recognize the next batch files,please close the window first!\n in the future,the client may be support multiple threads to improve the user experience"
save_to_csv_prompt="save result to a csv file"

emotion_field="emotion"
proba_field="proba"
show_confusion_matrix="show confusion matrix"
welcome="WelcomeUser"
main_page="MainPage"
analyzer="Analyzer"
settings="Settings"
about="about"

OK="OK"
clear_history="Clear History"
set_theme="Set Theme"
theme_prompt="See how elements look under different themes by choosing a different theme here!"

train_score="train_score"
test_score="test_score"
start_train="start train"
pca_components_tooltip = """
PCA components
Number of components to keep. if n_components is not set all 
components are kept:

n_components == min(n_samples, n_features)
If n_components == 'mle' and svd_solver == 'full', Minka’s MLE 
is used to guess the dimension. Use of n_components == 'mle' will 
interpret svd_solver == 'auto' as svd_solver == 'full'.

If 0 < n_components < 1 and svd_solver == 'full', select 
the number of components such that the amount of variance that 
needs to be explained is greater than the percentage specified 
by n_components.

If svd_solver == 'arpack', the number of components must
 be strictly less than the minimum of n_features and n_samples.

Hence, the None case results in:

n_components == min(n_samples, n_features) - 1
"""
pca_svd_solver_tooltip = """
If auto :
The solver is selected by a default policy based on X.shape and 
n_components:
 if the input data is larger than 500x500 and the number of components
   to extract is lower than 80% of the smallest dimension of the data, 
   then the more efficient ‘randomized’ method is enabled. Otherwise 
   the exact full SVD is computed and optionally truncated afterwards.

If full :
run exact full SVD calling the standard LAPACK solver via scipy.
linalg.svd and select the components by postprocessing

If arpack :
run SVD truncated to n_components calling ARPACK solver via 
scipy.sparse.linalg.svds. It requires strictly
 0 < n_components < min(X.shape)

If randomized :
run randomized SVD by the method of Halko et al.
"""
file_browse="BrosweFile"
fold_field="fold"
accu_score_field="accu_score"
application_menu="&Application"
exit_menu="E&xit"
help_menu="Help"
introduction_menu="Introduction"
predict_proba_legend="predict_proba"
current_model_prompt="current model:"
no_result_yet = f"No Result Yet"
language_switch="Switch"
audio_viewer="Audio Viewer"
print_display_prompt="Anything printed will display here!"
generate_pie_graph="Generate pie graph"
query_parameter_legend="Query usage records"
user_name="Username:"
corpus="Corpus:"
emotion_feature_prompt="Emotion Feature:"
recognition_alogs_prompt="Recognition Algorithm:"
recognized_file="Recognized File:"
emotion_compositon_analyzer_title="emotion composition analyzer"
select_audios_prompt="please select several files from the fviewer frist!😂"
save_to_file=f"save to file"
table_viewer_title="Table Viewer"
file_size="File Size"
files_count_unit=" files"
filter_options="filter_options"
audios_chooser="audios_chooser"
user='User'
register="Register"
login="Login"
welcome_login_title="Welcome to CCSER client"
input_register_info_prompt="Please enter your registration information:"
password="password:"
confirm_password="Confirm Password"
cancel="Cancel"
input_login_info_prompt="Please enter your login information:"
reset="Reset"