import sqlite3
import PySimpleGUI as sg

class UserAuthenticator:
    def __init__(self, db_file='users.db', verbose=False):
        self.db_file = db_file
        self.verbose = verbose
        self.conn = None
        self.c = None
    
    def connect_to_database(self):
        """连接SQLite数据库"""
        self.conn = sqlite3.connect(self.db_file)
        self.c = self.conn.cursor()
        if self.verbose:
            print(f"Connected to {self.db_file} database.")
    
    def close_database_connection(self):
        """关闭SQLite数据库连接"""
        self.c.close()
        self.conn.close()
        if self.verbose:
            print(f"Closed {self.db_file} database connection.")
    
    def create_users_table(self):
        """创建users表"""
        self.c.execute('''CREATE TABLE IF NOT EXISTS users
                          (username TEXT PRIMARY KEY, password TEXT)''')
        if self.verbose:
            print("Created users table.")
    
    def register_user(self, username, password, confirm_password):
        """注册用户"""
        if password == confirm_password:
            try:
                self.c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
                self.conn.commit()
                if self.verbose:
                    print(f"Registered user {username}.")
                return True
            except sqlite3.IntegrityError:
                if self.verbose:
                    print(f"Username {username} already exists. Please choose another username.")
                return False
        else:
            if self.verbose:
                print("Passwords do not match. Please try again.")
            return False
    
    def authenticate_user(self, username, password):
        """验证用户"""
        self.c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        result = self.c.fetchone()
        if result:
            if self.verbose:
                print(f"User {username} authenticated.")
            return True
        else:
            if self.verbose:
                print("Incorrect username or password. Please try again.")
            return False

class UserAuthenticatorGUI:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.authenticator = UserAuthenticator(verbose=self.verbose)
        self.layout = None
        self.window = None
    
    def create_layout(self):
        # 定义主题
        sg.theme('DarkTeal2')
        """创建PySimpleGUI窗口布局"""
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
        self.layout = [[sg.TabGroup([[sg.Tab('Register', register_layout), sg.Tab('Login', login_layout)]])]]
    
    def create_window(self):
        """创建PySimpleGUI窗口"""
        self.window = sg.Window('My App', self.layout)
    
    def start_event_loop(self):
        """开始PySimpleGUI事件循环"""
        while True:
            event, values = self.window.read()
            if event in (sg.WIN_CLOSED, 'Cancel'):
                break
            elif event == 'Register':
                username = values['username']
                password = values['password']
                confirm_password = values['confirm_password']
                if self.authenticator.register_user(username, password, confirm_password):
                    sg.popup(f'User {username} registered successfully!')
                else:
                    sg.popup('Registration failed. Please try again.')
            elif event == 'Login':
                username = values['username']
                password = values['password']
                if self.authenticator.authenticate_user(username, password):
                    sg.popup(f'User {username} authenticated successfully!')
                else:
                    sg.popup('Authentication failed. Please try again.')
    
    def run(self):
        """运行用户验证程序"""
        self.authenticator.connect_to_database()
        self.authenticator.create_users_table()
        self.create_layout()
        self.create_window()
        self.start_event_loop()
        self.authenticator.close_database_connection()

if __name__ == '__main__':
    gui = UserAuthenticatorGUI(verbose=True)
    gui.run()
