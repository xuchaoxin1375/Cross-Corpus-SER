import PySimpleGUI as sg

# 定义主题
sg.theme('DarkTeal2')

# 定义注册Tab页布局
register_layout = [[sg.Text('Welcome to My App', font=('Helvetica', 20))],
                   [sg.Text('Please enter your registration information:')],
                   [sg.Text('Username', size=(15, 1)), sg.InputText()],
                   [sg.Text('Password', size=(15, 1)), sg.InputText(password_char='*')],
                   [sg.Text('Confirm Password', size=(15, 1)), sg.InputText(password_char='*')],
                   [sg.Submit(button_text='Register'), sg.Cancel(button_text='Cancel')]]

# 定义登录Tab页布局
login_layout = [[sg.Text('Welcome to My App', font=('Helvetica', 20))],
                [sg.Text('Please enter your login information:')],
                [sg.Text('Username', size=(15, 1)), sg.InputText()],
                [sg.Text('Password', size=(15, 1)), sg.InputText(password_char='*')],
                [sg.Submit(button_text='Login'), sg.Cancel(button_text='Cancel')]]

# 定义Tab页布局
tab_layout = [[sg.TabGroup([[sg.Tab('Register', register_layout), sg.Tab('Login', login_layout)]])]]

# 创建窗口
window = sg.Window('My App', tab_layout)

# 事件循环
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break
    elif event == 'Register':
        username = values[0]
        password = values[1]
        confirm_password = values[2]
        if password == confirm_password:
            sg.popup('Registration successful!')
        else:
            sg.popup('Passwords do not match. Please try again.')
    elif event == 'Login':
        username = values[3]
        password = values[4]
        sg.popup('Login successful!')

window.close()
