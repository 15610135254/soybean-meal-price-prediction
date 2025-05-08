from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from views.auth import login_required
import os
import json
import logging
from datetime import datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个名为 'news' 的 Blueprint
bp = Blueprint('news', __name__)

# 新闻数据文件路径 (相对于当前文件)
NEWS_FILE_PATH = '../data/news.json'

# 新闻分类
NEWS_CATEGORIES = [
    {"id": "all", "name": "全部", "active": True},
    {"id": "market", "name": "市场动态", "active": False},
    {"id": "policy", "name": "政策法规", "active": False},
    {"id": "industry", "name": "行业分析", "active": False},
    {"id": "international", "name": "国际资讯", "active": False},
    {"id": "company", "name": "企业新闻", "active": False}
]

# 模拟新闻数据
SAMPLE_NEWS = [
    {
        "id": 1,
        "title": "中国大豆进口量创新高，豆粕价格承压",
        "content": "据海关总署数据显示，今年第一季度中国大豆进口量同比增长15.2%，创历史新高。分析师认为，进口量增加将导致豆粕供应充足，短期内价格可能承压。",
        "category": "market",
        "category_name": "市场动态",
        "image": "https://via.placeholder.com/1200x400",
        "date": "2023-05-03",
        "source": "农业经济网",
        "views": 3456,
        "is_featured": True
    },
    {
        "id": 2,
        "title": "全球大豆产量预测报告发布，市场波动加剧",
        "content": "美国农业部发布最新全球大豆产量预测报告，预计本年度全球大豆产量将达到3.68亿吨，较上年增长2.5%。报告发布后，芝加哥期货交易所大豆期货价格出现波动。",
        "category": "market",
        "category_name": "市场动态",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-05-02",
        "source": "农业经济网",
        "views": 2345,
        "is_featured": False
    },
    {
        "id": 3,
        "title": "饲料需求增长，豆粕期货价格有望上涨",
        "content": "随着畜牧业复苏和养殖规模扩大，国内饲料需求持续增长。分析师预计，下半年豆粕需求将进一步提升，期货价格有望走高。",
        "category": "industry",
        "category_name": "行业分析",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-05-01",
        "source": "期货日报",
        "views": 1987,
        "is_featured": False
    },
    {
        "id": 4,
        "title": "巴西大豆收获进度超预期，出口量创新高",
        "content": "巴西农业部数据显示，截至4月底，巴西大豆收获进度已达95%，高于去年同期的92%。同时，一季度大豆出口量达到2250万吨，同比增长12%，创历史新高。",
        "category": "international",
        "category_name": "国际资讯",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-04-30",
        "source": "国际商报",
        "views": 1756,
        "is_featured": False
    },
    {
        "id": 5,
        "title": "农业农村部发布大豆振兴计划，提高国内大豆自给率",
        "content": "农业农村部近日发布《大豆振兴计划实施方案》，提出到2025年大豆种植面积稳定在1.4亿亩以上，自给率提高到40%以上的目标。",
        "category": "policy",
        "category_name": "政策法规",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-04-28",
        "source": "中国农业新闻网",
        "views": 1543,
        "is_featured": False
    },
    {
        "id": 6,
        "title": "中美贸易关系改善，大豆进口政策或将调整",
        "content": "近日，中美两国代表举行会谈，就农产品贸易问题达成初步共识。分析人士认为，这可能促使中国调整大豆进口政策，增加美国大豆的进口量。",
        "category": "policy",
        "category_name": "政策法规",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-04-27",
        "source": "国际商报",
        "views": 2345,
        "is_featured": False
    },
    {
        "id": 7,
        "title": "豆粕期货主力合约创年内新高，后市如何走？",
        "content": "昨日，大连商品交易所豆粕期货主力合约盘中一度触及3500元/吨，创年内新高。多位分析师认为，在供需格局改善的背景下，豆粕期货价格仍有上涨空间。",
        "category": "market",
        "category_name": "市场动态",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-04-25",
        "source": "期货日报",
        "views": 1987,
        "is_featured": False
    },
    {
        "id": 8,
        "title": "深度分析：气候变化对全球大豆产量的影响",
        "content": "近年来，全球气候变化加剧，极端天气事件频发，对农业生产造成严重影响。本文深入分析气候变化对全球大豆产量的影响，并探讨应对策略。",
        "category": "industry",
        "category_name": "行业分析",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-04-23",
        "source": "农业科技报",
        "views": 1756,
        "is_featured": False
    },
    {
        "id": 9,
        "title": "豆粕期货交易技巧：如何把握季节性规律",
        "content": "豆粕期货价格存在明显的季节性波动规律。本文分析近十年豆粕期货价格数据，总结季节性规律，并提出相应的交易策略建议。",
        "category": "industry",
        "category_name": "行业分析",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-04-20",
        "source": "期货日报",
        "views": 1543,
        "is_featured": False
    },
    {
        "id": 10,
        "title": "国内大豆压榨企业产能利用率提升至85%",
        "content": "据行业协会统计，截至4月中旬，国内大豆压榨企业平均产能利用率已提升至85%，较去年同期提高10个百分点，表明下游需求正在恢复。",
        "category": "company",
        "category_name": "企业新闻",
        "image": "https://via.placeholder.com/300x200",
        "date": "2023-04-18",
        "source": "中国粮油信息网",
        "views": 1298,
        "is_featured": False
    }
]

# 市场日历数据
MARKET_CALENDAR = [
    {
        "id": 1,
        "title": "USDA大豆产量报告",
        "organization": "美国农业部",
        "date": "2023-05-10"
    },
    {
        "id": 2,
        "title": "豆粕期货交割日",
        "organization": "大连商品交易所",
        "date": "2023-05-15"
    },
    {
        "id": 3,
        "title": "全球油脂油料峰会",
        "organization": "北京",
        "date": "2023-05-20"
    }
]

def load_news_data():
    """加载新闻数据"""
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        news_file_path = os.path.join(current_dir, NEWS_FILE_PATH)

        # 检查文件是否存在
        if os.path.exists(news_file_path):
            with open(news_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"新闻数据文件不存在: {news_file_path}，使用示例数据")
            return SAMPLE_NEWS
    except Exception as e:
        logger.error(f"加载新闻数据时出错: {e}")
        return SAMPLE_NEWS

def save_news_data(news_data):
    """保存新闻数据"""
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        news_file_path = os.path.join(current_dir, NEWS_FILE_PATH)

        # 确保目录存在
        os.makedirs(os.path.dirname(news_file_path), exist_ok=True)

        with open(news_file_path, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=4)

        logger.info(f"新闻数据已保存到: {news_file_path}")
        return True
    except Exception as e:
        logger.error(f"保存新闻数据时出错: {e}")
        return False

@bp.route('/')
@login_required
def index():
    """新闻资讯首页（需要登录）"""
    # 获取分类过滤参数
    category = request.args.get('category', 'all')

    # 加载新闻数据
    news_data = load_news_data()

    # 根据分类过滤新闻
    if category != 'all':
        filtered_news = [news for news in news_data if news['category'] == category]
    else:
        filtered_news = news_data

    # 获取头条新闻（标记为featured的新闻）
    featured_news = next((news for news in news_data if news.get('is_featured')), news_data[0] if news_data else None)

    # 获取热门新闻（按浏览量排序）
    popular_news = sorted(news_data, key=lambda x: x.get('views', 0), reverse=True)[:5]

    # 更新分类的激活状态
    categories = NEWS_CATEGORIES.copy()
    for cat in categories:
        cat['active'] = (cat['id'] == category)

    return render_template(
        'news/index.html',
        news_list=filtered_news,
        featured_news=featured_news,
        popular_news=popular_news,
        categories=categories,
        market_calendar=MARKET_CALENDAR,
        current_category=category
    )

@bp.route('/detail/<int:news_id>')
@login_required
def detail(news_id):
    """新闻详情页（需要登录）"""
    # 加载新闻数据
    news_data = load_news_data()

    # 查找指定ID的新闻
    news_item = next((news for news in news_data if news['id'] == news_id), None)

    if not news_item:
        # 如果找不到新闻，返回404
        return render_template('404.html'), 404

    # 增加浏览量
    news_item['views'] = news_item.get('views', 0) + 1
    save_news_data(news_data)

    # 获取相关新闻（同类别的其他新闻）
    related_news = [
        news for news in news_data
        if news['category'] == news_item['category'] and news['id'] != news_id
    ][:3]

    return render_template(
        'news/detail.html',
        news=news_item,
        related_news=related_news
    )

@bp.route('/search')
def search():
    """搜索新闻"""
    # 获取搜索关键词
    keyword = request.args.get('keyword', '')

    if not keyword:
        return jsonify({'error': '请输入搜索关键词'}), 400

    # 加载新闻数据
    news_data = load_news_data()

    # 搜索标题和内容中包含关键词的新闻
    search_results = [
        news for news in news_data
        if keyword.lower() in news['title'].lower() or keyword.lower() in news['content'].lower()
    ]

    return render_template(
        'news/search.html',
        news_list=search_results,
        keyword=keyword,
        count=len(search_results)
    )

@bp.route('/api/subscribe', methods=['POST'])
def subscribe():
    """订阅新闻通讯"""
    email = request.form.get('email')

    if not email:
        return jsonify({'success': False, 'message': '请输入有效的邮箱地址'}), 400

    # 这里应该有保存订阅者邮箱的逻辑
    # 简单起见，我们只返回成功消息

    return jsonify({
        'success': True,
        'message': f'感谢订阅！我们会将最新资讯发送到 {email}'
    })
