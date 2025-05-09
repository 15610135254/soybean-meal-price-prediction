from flask import Blueprint, render_template, redirect, url_for, session


bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """首页路由，检查用户是否已登录，未登录则重定向到登录页面"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('index.html')

