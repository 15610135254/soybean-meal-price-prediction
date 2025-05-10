import requests
from bs4 import BeautifulSoup
import json
import time
import sys
import re
import os
import random
from datetime import datetime
from urllib.parse import urljoin

def extract_publish_time(container, url):
    if container:
        time_selectors = [
            '.time', '.date', '.publish-time', '.article-time', '.news-time',
            'time', '.timestamp', '.pubtime', '.publish-date', '.news-date'
        ]

        for selector in time_selectors:
            time_tag = container.select_one(selector)
            if time_tag:
                time_text = time_tag.get_text(strip=True)
                if time_text:
                    return time_text

        text = container.get_text()

        date_patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?(\s\d{1,2}:\d{1,2}(:\d{1,2})?)?',
            r'\d{1,2}[-/月]\d{1,2}[日]?(\s\d{1,2}:\d{1,2}(:\d{1,2})?)?',
            r'\d{1,2}:\d{1,2}(:\d{1,2})?'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

    if url:
        url_date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        match = re.search(url_date_pattern, url)
        if match:
            return match.group(0)

    return ""

def get_news_content(url, title, max_retries=2):
    """获取新闻内容"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }

    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '').lower()
            if 'charset=' in content_type:
                detected_encoding = content_type.split('charset=')[-1].strip()
                if detected_encoding:
                    encodings.insert(0, detected_encoding)

            html_text = None
            for encoding in encodings:
                try:
                    html_text = response.content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if html_text is None:
                html_text = response.text

            soup = BeautifulSoup(html_text, 'html.parser')

            content_selectors = [
                'article', '.article-content', '.news-content', '.content',
                '.article-body', '.news-text', '.article-text', '.news-detail',
                '.article', '#article', '.main-content', '.main-article'
            ]

            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    paragraphs = content_element.find_all('p')
                    if paragraphs:
                        valid_paragraphs = []
                        for p in paragraphs:
                            text = p.get_text(strip=True)
                            if text and len(text) > 15:
                                valid_paragraphs.append(text)
                            if len(valid_paragraphs) >= 3:
                                break

                        if valid_paragraphs:
                            content = ' '.join(valid_paragraphs)
                            if not is_gibberish(content):
                                return content[:500]

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content = meta_desc.get('content')
                if not is_gibberish(content):
                    return content[:500]

            body_text = soup.body.get_text(strip=True) if soup.body else ""
            if body_text and len(body_text) > 50 and not is_gibberish(body_text):
                sentences = body_text.split('。')
                content = '。'.join(sentences[:3]) + '。' if sentences else body_text[:500]
                return content[:500]

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"获取新闻内容失败: {str(e)}")

    return f"{title}。这是一条关于豆粕期货市场的重要新闻。"

# 检查文本是否为乱码
def is_gibberish(text):
    if not text:
        return True

    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / len(text) > 0.5:
        return True

    if text.count('') > 5 or text.count('?') > len(text) * 0.2:
        return True

    if any('\u4e00' <= c <= '\u9fff' for c in text):
        if text.count('。') == 0 and text.count('，') == 0 and text.count('、') == 0:
            return True

    return False

def get_news_info(url, max_retries=3, retry_delay=2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            if response.encoding == 'ISO-8859-1':
                response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            raw_news_list = []
            news_items = soup.select('a.news-title-font_1xS-F')

            if news_items:
                for item in news_items:
                    link = item.get('href')
                    title = item.get_text(strip=True)
                    if link:
                        publish_time = extract_publish_time(item.parent, link)

                        if not publish_time:
                            publish_time = datetime.now().strftime('%Y-%m-%d')

                        raw_news = {
                            'url': link,
                            'title': title if title else '无标题',
                            'publish_time': publish_time
                        }
                        raw_news_list.append(raw_news)
            else:
                news_containers = soup.select('.news-item, .article-item, .news-box, .news-list li, .article-list li')

                if news_containers:
                    for container in news_containers:
                        link_tag = container.find('a')
                        if link_tag:
                            href = link_tag.get('href')
                            if href and not href.startswith('javascript:') and not href.startswith('#'):
                                full_url = urljoin(url, href)
                                title = link_tag.get_text(strip=True)
                                if not title:
                                    title_tag = container.find(['h1', 'h2', 'h3', 'h4', 'h5', '.title', '.news-title'])
                                    if title_tag:
                                        title = title_tag.get_text(strip=True)

                                publish_time = extract_publish_time(container, full_url)

                                if not publish_time:
                                    publish_time = datetime.now().strftime('%Y-%m-%d')

                                raw_news = {
                                    'url': full_url,
                                    'title': title if title else '无标题',
                                    'publish_time': publish_time
                                }
                                raw_news_list.append(raw_news)
                else:
                    all_links = soup.find_all('a')
                    for link in all_links:
                        href = link.get('href')
                        if href and not href.startswith('javascript:') and not href.startswith('#'):
                            full_url = urljoin(url, href)
                            title = link.get_text(strip=True)

                            news_keywords = ['news', 'article', 'detail', 'content', 'story', 'report']
                            exclude_keywords = ['login', 'register', 'about', 'contact', 'help', 'download', 'app']

                            is_news_link = any(keyword in full_url.lower() for keyword in news_keywords)
                            is_excluded = any(keyword in full_url.lower() for keyword in exclude_keywords)

                            has_valid_title = title and len(title) > 5

                            if is_news_link and not is_excluded and has_valid_title:
                                publish_time = extract_publish_time(link.parent, full_url)

                                if not publish_time:
                                    publish_time = datetime.now().strftime('%Y-%m-%d')

                                raw_news = {
                                    'url': full_url,
                                    'title': title,
                                    'publish_time': publish_time
                                }
                                raw_news_list.append(raw_news)

            news_info_list = []
            for i, news in enumerate(raw_news_list):
                content = get_news_content(news['url'], news['title'])

                if is_gibberish(content):
                    content = f"{news['title']}。这是一条关于豆粕期货市场的重要新闻。"
                category = "market"
                category_name = "市场动态"
                is_featured = (i == 0)
                views = random.randint(1000, 3500)
                source = "网络"
                news_info = {
                    'id': i + 1,
                    'title': news['title'],
                    'content': content,
                    'category': category,
                    'category_name': category_name,
                    'date': news['publish_time'],
                    'source': source,
                    'views': views,
                    'is_featured': is_featured,
                    'url': news['url']
                }
                news_info_list.append(news_info)

            return news_info_list

        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return []
        except Exception as e:
            print(f"获取新闻时出错: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return []

def save_to_file(data, filename):
    """保存数据到文件"""
    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"数据已保存到: {filename}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return False

def print_usage():
    """显示使用说明"""
    print("使用方法:")
    print("python3 spider.py [URL] [输出文件名]")
    print("示例:")
    print("python3 spider.py https://news.sina.com.cn/ data/news.json")
    print("如果不提供参数，将使用默认URL(百度豆粕新闻搜索)和默认输出文件名(data/news.json)")

def main():
    url = None
    output_file = 'data/news.json'

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_usage()
            return

        url = sys.argv[1]

        if len(sys.argv) > 2:
            output_file = sys.argv[2]
            if not output_file.endswith('.json'):
                output_file += '.json'
    else:
        url = 'https://www.baidu.com/s?tn=news&rtt=1&bsst=1&wd=%E8%B1%86%E7%B2%95%E6%96%B0%E9%97%BB&cl=1'

    print(f"正在从 {url} 爬取新闻...")
    news_info_list = get_news_info(url)

    if news_info_list:
        print(f"成功获取 {len(news_info_list)} 条新闻")

        if save_to_file(news_info_list, output_file):
            print(f"新闻数据已保存到 {output_file}")

            if output_file != 'news_info.json':
                save_to_file(news_info_list, 'news_info.json')
    else:
        print("未找到任何新闻链接")

if __name__ == "__main__":
    main()