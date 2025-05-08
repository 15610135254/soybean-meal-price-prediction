from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
import os
import json
import logging
from functools import wraps

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个名为 'auth' 的 Blueprint
bp = Blueprint('auth', __name__)

# 用户数据文件路径 (相对于当前文件)
USERS_FILE = '../data/forum/users.json'

def load_users():
    """加载用户数据"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, USERS_FILE)
        
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"用户数据文件不存在: {full_path}")
            return []
    except Exception as e:
        logger.error(f"加载用户数据时出错: {str(e)}")
        return []

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

@bp.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')  # 实际应用中应该使用加密密码
        
        # 简单起见，这里不验证密码，只检查用户名是否存在
        users = load_users()
        user = next((u for u in users if u['username'] == username), None)
        
        if user:
            # 登录成功，保存用户信息到会话
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_role'] = user['role']
            
            flash(f'欢迎回来，{username}！', 'success')
            return redirect(url_for('main.index'))
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
