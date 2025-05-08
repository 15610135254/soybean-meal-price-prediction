from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
import os
import json
import logging
from functools import wraps
from datetime import datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个名为 'auth' 的 Blueprint
bp = Blueprint('auth', __name__)

# 用户数据文件路径 (相对于当前文件)
USERS_FILE = '../data/forum/users.json'

def get_users_file_path():
    """获取用户数据文件的完整路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, USERS_FILE)

def load_users():
    """加载用户数据"""
    try:
        full_path = get_users_file_path()

        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"用户数据文件不存在: {full_path}")
            return []
    except Exception as e:
        logger.error(f"加载用户数据时出错: {str(e)}")
        return []

def save_users(users):
    """保存用户数据"""
    try:
        full_path = get_users_file_path()

        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4, ensure_ascii=False)

        logger.info(f"用户数据已保存到: {full_path}")
        return True
    except Exception as e:
        logger.error(f"保存用户数据时出错: {str(e)}")
        return False

def login_required(view):
    """登录验证装饰器"""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return view(*args, **kwargs)
    return wrapped_view

def admin_required(view):
    """管理员权限验证装饰器"""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))

        if session.get('user_role') != 'admin':
            flash('您没有权限访问该页面', 'danger')
            return redirect(url_for('main.index'))

        return view(*args, **kwargs)
    return wrapped_view

@bp.route('/register', methods=['GET', 'POST'])
def register():
    """注册页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')  # 实际应用中应该使用加密密码
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'user')  # 默认为普通用户

        # 验证输入
        if not username or not password:
            flash('用户名和密码不能为空', 'danger')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('两次输入的密码不一致', 'danger')
            return render_template('auth/register.html')

        # 检查用户名是否已存在
        users = load_users()
        if any(u['username'] == username for u in users):
            flash('用户名已存在，请选择其他用户名', 'danger')
            return render_template('auth/register.html')

        # 创建新用户
        new_user = {
            "id": max([u['id'] for u in users], default=0) + 1,
            "username": username,
            "avatar": "https://via.placeholder.com/40",
            "role": role,
            "posts": 0,
            "replies": 0,
            "join_date": datetime.now().strftime('%Y-%m-%d'),
            "last_active": datetime.now().strftime('%Y-%m-%d')
        }

        # 添加新用户并保存
        users.append(new_user)
        if save_users(users):
            flash('注册成功，请登录', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('注册失败，请稍后再试', 'danger')

    return render_template('auth/register.html')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')  # 实际应用中应该使用加密密码
        role = request.form.get('role')  # 获取选择的角色

        # 加载用户数据
        users = load_users()

        # 根据角色筛选用户
        if role:
            filtered_users = [u for u in users if u['role'] == role]
        else:
            filtered_users = users

        # 查找用户
        user = next((u for u in filtered_users if u['username'] == username), None)

        if user:
            # 登录成功，保存用户信息到会话
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_role'] = user['role']

            flash(f'欢迎回来，{username}！', 'success')
            return redirect(url_for('main.index'))
        else:
            if role:
                flash(f'未找到{role}角色的用户"{username}"或密码错误', 'danger')
            else:
                flash('用户名或密码错误', 'danger')

    return render_template('auth/login.html')

@bp.route('/logout')
def logout():
    """登出"""
    session.clear()
    flash('您已成功登出', 'success')
    return redirect(url_for('main.index'))

@bp.route('/api/check-auth')
def check_auth():
    """API端点：检查用户是否已登录"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session['user_id'],
                'username': session['username'],
                'role': session['user_role']
            }
        })
    else:
        return jsonify({'authenticated': False})
