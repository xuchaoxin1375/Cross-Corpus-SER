import sqlite3
import PySimpleGUI as sg

# 创建数据库连接
conn = sqlite3.connect('users.db')
c = conn.cursor()

# 创建users表
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')

# 定义主题
sg.theme('DarkTeal2')

# 定义注册Tab页布局
register_layout = [[sg.Text('Welcome to My App', font=('Helvetica', 20))],
                   [sg.Text('Please enter your registration information:')],
                   [sg.Text('Username', size=(15, 1)), sg.InputText(key='username')],
                   [sg.Text('Password', size=(15, 1)), sg.InputText(key='password', password_char='*')],
                   [sg.Text('Confirm Password', size=(15, 1)), sg.InputText(key='confirm_password', password_char='*')],
                   [sg.Submit(button_text='Register'), sg.Cancel(button_text='Cancel')]]

# 定义登录Tab页布局
login_layout = [[sg.Text('Welcome to My App', font=('Helvetica', 20))],
                [sg.Text('Please enter your login information:')],
                [sg.Text('Username', size=(15, 1)), sg.InputText(key='username')],
                [sg.Text('Password', size=(15, 1)), sg.InputText(key='password', password_char='*')],
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
        username = values['username']
        password = values['password']
        confirm_password = values['confirm_password']
        if password == confirm_password:
            # 将用户名和密码保存到数据库中
            try:
                c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
                conn.commit()
                sg.popup('Registration successful!')
            except sqlite3.IntegrityError:
                sg.popup('Username already exists. Please choose another username.')
        else:
            sg.popup('Passwords do not match. Please try again.')
    elif event == 'Login':
        username = values['username']
        password = values['password']
        # 查询数据库中是否存在对应的用户名和密码
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
        if result:
            sg.popup('Login successful!')
        else:
            sg.popup('Incorrect username or password. Please try again.')

# 关闭数据库连接和窗口
c.close()
conn.close()
window.close()
