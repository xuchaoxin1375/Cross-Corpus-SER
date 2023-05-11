import SG.constants.logo as logo

welcome_message = "欢迎来到我的应用程序！"
choose_emotion_config = """
请选择一个情感组合进行测试：建议使用AS、HNS、AHNS、AHNPS组合。请注意，在SAVEE数据集中，
“surprise”和“pleasantSurprise”之间存在差异，因此不建议在SAVEE上使用AHNPS组合。
"""
choose_feature_config = "请选择一个或多个特征"
feature_transform_config = "特征变换器配置"
choose_algorithm = "选择一个算法进行测试"
choose_audio = "请选择一个音频样本文件以识别其情感"
recognize_the_audio_emotion = "识别所选音频文件的情感"
draw_diagram = "选择一个图表进行绘制"
draw_diagram_detail = "绘制所选文件的[波形图|频谱图|Mel频谱图]："
welcome_title = "𝒲ℯ𝓁𝒸ℴ𝓂ℯ 𝓉ℴ ℯ𝓍𝓅ℯ𝓇𝒾ℯ𝓃𝒸ℯ 𝒞𝒞𝒮ℰℛ 𝒞𝓁𝒾ℯ𝓃𝓉!"
result_training = "模型训练结果："
train_result_title = "训练结果"
result_frame_prompt = "所选文件的情感（预测结果）"
recognize_it = "开始识别"
WaveForm = "波形图"
FreqGraph = "频谱图"
MelFreqGraph = "Mel频谱图"
algorithmes_chooser_title = "算法选择器"
choose_emotion_config_title = "情感配置选择器"
emotion_config_legend = "选择情感配置"
feature_transform_legend = "特征变换选择器"
feature_config_legend = "特征配置选择器"

n_components_tooltip = "输入要保留的成分数。"
n_components_prompt = "主成分数："
demension_prompt = "特征维数："
pending_prompt = "待处理"

other_parameter_legend = "其他参数设置"
select_training_db = "选择训练数据库"
select_testing_db = "选择测试数据库"
files_selected_prompt = "已选择的文件："
confirm_files_selected_button = "确认所选文件"
recursively_scan_subdir = "递归扫描子目录"
auto_refresh = "自动刷新"
auto_refresh_tooltip = "点击手动刷新文件列表框"
short_path = "短路径模式"
filter_by_regex_prompt = "使用正则表达式过滤："
selected_audios_prompt = "已选择的音频文件："
no_files = "0 个文件"
audios_filter = "音频文件过滤"
filterd_audios = "过滤后的音频文件："
files_count_unit = " 个文件"
train_model_warning = "请先训练SER模型，然后再试一次！"

selected_files_tooltip = """
您可以观察上一个列表框中选择的文件，无论是连续选择还是跳跃选择，这些选择的文件都将被紧密排列，并在顶部显示文件数量。
"""
filter_tooltip = """
文件列表框允许您使用鼠标的左键选择一个或多个文件，您可以使用Ctrl +单击选择多个文件（还可以跳转到所选文件！）

您可以选择一个或多个文件后右键单击以执行以下操作：
1.文件大小
2.文件路径（绝对路径）
3.识别情感
4.播放所选文件（音频）
*.以上所有内容都可以在多个文件中自动一一工作
"""
not_exist = "不存在！"
confirm_folder_selected = "确认已选择的文件夹"
folder_browse = "浏览文件夹"
choose_folder_tooltip = "选择一个要进行SER的文件夹， 默认文件夹为 "
files_browse = "浏览文件"
current_directory_prompt = "当前目录："
show_file_path = "显示文件路径"
show_file_size = "显示文件大小"
play_audio = "播放音频"
emotion_recognize = "情感识别"
path_input_tooltip = "您可以粘贴或输入目录路径！或使用右侧的浏览按钮选择目录"
click_filter_prompt = "点击过滤器或输入正则表达式来扫描音频文件！"
listbox_default_value_prompt = "将鼠标悬停在此列表框区域以查看工具提示！"
filter_audios = "过滤音频文件"
select_dir_prompt = "选择语音库目录以浏览语音文件"

save_patient_warning = "点击保存到CSV文件！保存操作将花费一些时间来完成！请耐心等待！"
close_table_prompt = "如果您想识别下一批文件，请先关闭窗口！在未来，客户端可能支持多线程以提高用户体验"
save_to_csv_prompt = "将结果保存到CSV文件"

emotion_field = "情感"
proba_field = "概率"
show_confusion_matrix = "显示混淆矩阵"
welcome = "欢迎"
main_page = "主页"
analyzer = "分析器"
settings = "设置"
about = "关于"

OK = "确定"
clear_history = "清除历史记录"
set_theme = "设置主题"
theme_prompt = "通过在此处选择不同的主题来查看元素在不同主题下的外观！"

train_score = "训练集得分"
test_score = "测试集得分"
start_train = "开始训练"
pca_components_tooltip = """
PCA组件
要保留的组件数。如果未设置n_components，则保留所有组件：

n_components == min(n_samples, n_features)
如果n_components == 'mle'且svd_solver =='full'，则使用Minka的MLE猜测维数。使用n_components =='mle'将解释为svd_solver =='auto'，即svd_solver =='full'。

如果0 < n_components < 1且svd_solver =='full'，则选择组件数，使需要解释的方差量大于n_components指定的百分比。

如果svd_solver =='arpack'，则组件数必须严格小于n_features和n_samples的最小值。

因此，None情况的结果为：

n_components == min(n_samples, n_features) - 1
"""
pca_svd_solver_tooltip = """
如果自动:
根据X.shape和n_components选择默认策略选择求解器：
如果输入数据大于500x500并且提取的组件数小于数据的最小维度的80％，
则启用更有效的“随机”方法。否则，计算精确的全SVD，然后可选择截断。

如果完整：
通过scipy运行准确的完整SVD。linalg.svd并通过后处理选择组件

如果arpack：
通过scipy.sparse.linalg.svds调用截断为n_components的SVD ARPACK求解器。它需要 0 < n_components < min(X.shape)

如果随机：
通过Halko等人的方法运行随机SVD。
"""
file_browse = "浏览单个文件"

fold_field = "k-折"
accu_score_field = "准确度"
application_menu = "&应用程序"
exit_menu = "退出(&x)"
help_menu = "帮助"
introduction_menu = "介绍"
predict_proba_legend = "预测概率"
current_model_prompt = "当前模型："
no_result_yet = "暂无结果"
language_switch="确认切换"
audio_viewer="音频文件浏览器"
print_display_prompt="任何打印的内容(日志)将会输出于此:"
generate_pie_graph="生成饼图"
query_parameter_legend="使用记录查询"
user_name="用户名:"
corpus="语料库"
emotion_feature_prompt="情感特征"
recognition_alogs_prompt="识别算法"
recognized_file="识别的文件"
emotion_compositon_analyzer_title="情感组成分析"
select_audios_prompt="请从fviewer中选择若干文件识别后重试！😂"
save_to_file = "保存到文件"
table_viewer_title="表格视图"
file_size="文件大小"