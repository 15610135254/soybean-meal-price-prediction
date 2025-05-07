/**
 * 通用分页功能
 * 用于新闻资讯和论坛讨论的分页导航
 */

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化新闻分页
    initNewsPagination();
    
    // 初始化论坛分页
    initForumPagination();
});

/**
 * 初始化新闻分页
 */
function initNewsPagination() {
    // 获取新闻列表容器
    const newsContainer = document.querySelector('.news-list-container');
    if (!newsContainer) return;
    
    // 获取所有新闻卡片
    const newsCards = newsContainer.querySelectorAll('.news-card');
    if (newsCards.length === 0) return;
    
    // 每页显示的新闻数量
    const itemsPerPage = 5;
    
    // 总页数
    const totalPages = Math.ceil(newsCards.length / itemsPerPage);
    
    // 当前页码
    let currentPage = 1;
    
    // 获取分页导航
    const pagination = document.querySelector('.news-pagination');
    if (!pagination) return;
    
    // 更新分页导航
    updatePagination(pagination, currentPage, totalPages, 'news');
    
    // 显示第一页
    showPage(newsCards, currentPage, itemsPerPage);
    
    // 添加分页导航事件监听
    pagination.addEventListener('click', function(e) {
        if (e.target.tagName === 'A' || e.target.parentElement.tagName === 'A') {
            e.preventDefault();
            
            const pageItem = e.target.closest('.page-item');
            if (!pageItem) return;
            
            if (pageItem.classList.contains('disabled')) return;
            
            if (pageItem.classList.contains('page-prev')) {
                if (currentPage > 1) {
                    currentPage--;
                }
            } else if (pageItem.classList.contains('page-next')) {
                if (currentPage < totalPages) {
                    currentPage++;
                }
            } else {
                const pageNum = parseInt(e.target.textContent);
                if (!isNaN(pageNum)) {
                    currentPage = pageNum;
                }
            }
            
            // 更新分页导航
            updatePagination(pagination, currentPage, totalPages, 'news');
            
            // 显示当前页
            showPage(newsCards, currentPage, itemsPerPage);
            
            // 滚动到新闻列表顶部
            newsContainer.scrollIntoView({ behavior: 'smooth' });
        }
    });
}

/**
 * 初始化论坛分页
 */
function initForumPagination() {
    // 获取论坛主题列表容器
    const topicContainer = document.querySelector('.topic-list-container');
    if (!topicContainer) return;
    
    // 获取所有主题卡片
    const topicCards = topicContainer.querySelectorAll('.topic-card');
    if (topicCards.length === 0) return;
    
    // 每页显示的主题数量
    const itemsPerPage = 10;
    
    // 总页数
    const totalPages = Math.ceil(topicCards.length / itemsPerPage);
    
    // 当前页码
    let currentPage = 1;
    
    // 获取分页导航
    const pagination = document.querySelector('.forum-pagination');
    if (!pagination) return;
    
    // 更新分页导航
    updatePagination(pagination, currentPage, totalPages, 'forum');
    
    // 显示第一页
    showPage(topicCards, currentPage, itemsPerPage);
    
    // 添加分页导航事件监听
    pagination.addEventListener('click', function(e) {
        if (e.target.tagName === 'A' || e.target.parentElement.tagName === 'A') {
            e.preventDefault();
            
            const pageItem = e.target.closest('.page-item');
            if (!pageItem) return;
            
            if (pageItem.classList.contains('disabled')) return;
            
            if (pageItem.classList.contains('page-prev')) {
                if (currentPage > 1) {
                    currentPage--;
                }
            } else if (pageItem.classList.contains('page-next')) {
                if (currentPage < totalPages) {
                    currentPage++;
                }
            } else {
                const pageNum = parseInt(e.target.textContent);
                if (!isNaN(pageNum)) {
                    currentPage = pageNum;
                }
            }
            
            // 更新分页导航
            updatePagination(pagination, currentPage, totalPages, 'forum');
            
            // 显示当前页
            showPage(topicCards, currentPage, itemsPerPage);
            
            // 滚动到主题列表顶部
            topicContainer.scrollIntoView({ behavior: 'smooth' });
        }
    });
}

/**
 * 显示指定页的内容
 * @param {NodeList} items - 所有项目
 * @param {number} page - 当前页码
 * @param {number} itemsPerPage - 每页显示的项目数量
 */
function showPage(items, page, itemsPerPage) {
    // 计算当前页的起始和结束索引
    const startIndex = (page - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, items.length);
    
    // 隐藏所有项目
    items.forEach(item => {
        item.style.display = 'none';
    });
    
    // 显示当前页的项目
    for (let i = startIndex; i < endIndex; i++) {
        items[i].style.display = '';
    }
}

/**
 * 更新分页导航
 * @param {HTMLElement} pagination - 分页导航元素
 * @param {number} currentPage - 当前页码
 * @param {number} totalPages - 总页数
 * @param {string} type - 分页类型（'news' 或 'forum'）
 */
function updatePagination(pagination, currentPage, totalPages, type) {
    // 清空分页导航
    pagination.innerHTML = '';
    
    // 创建"上一页"按钮
    const prevItem = document.createElement('li');
    prevItem.className = `page-item page-prev ${currentPage === 1 ? 'disabled' : ''}`;
    const prevLink = document.createElement('a');
    prevLink.className = 'page-link';
    prevLink.href = '#';
    prevLink.textContent = '上一页';
    prevItem.appendChild(prevLink);
    pagination.appendChild(prevItem);
    
    // 创建页码按钮
    const maxPageButtons = 5; // 最多显示的页码按钮数
    const startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
    const endPage = Math.min(totalPages, startPage + maxPageButtons - 1);
    
    for (let i = startPage; i <= endPage; i++) {
        const pageItem = document.createElement('li');
        pageItem.className = `page-item ${i === currentPage ? 'active' : ''}`;
        const pageLink = document.createElement('a');
        pageLink.className = 'page-link';
        pageLink.href = '#';
        pageLink.textContent = i;
        pageItem.appendChild(pageLink);
        pagination.appendChild(pageItem);
    }
    
    // 创建"下一页"按钮
    const nextItem = document.createElement('li');
    nextItem.className = `page-item page-next ${currentPage === totalPages ? 'disabled' : ''}`;
    const nextLink = document.createElement('a');
    nextLink.className = 'page-link';
    nextLink.href = '#';
    nextLink.textContent = '下一页';
    nextItem.appendChild(nextLink);
    pagination.appendChild(nextItem);
    
    // 更新分页信息
    const infoElement = document.querySelector(`.${type}-pagination-info`);
    if (infoElement) {
        const startItem = (currentPage - 1) * (type === 'news' ? 5 : 10) + 1;
        const endItem = Math.min(currentPage * (type === 'news' ? 5 : 10), type === 'news' ? document.querySelectorAll('.news-card').length : document.querySelectorAll('.topic-card').length);
        const totalItems = type === 'news' ? document.querySelectorAll('.news-card').length : document.querySelectorAll('.topic-card').length;
        infoElement.textContent = `显示 ${startItem}-${endItem} 条，共 ${totalItems} 条`;
    }
}
