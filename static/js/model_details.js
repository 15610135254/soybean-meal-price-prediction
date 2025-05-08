/**
 * 模型详情处理脚本
 * 用于加载和显示模型架构信息
 */

document.addEventListener('DOMContentLoaded', function() {
    // 初始化模型详情对话框
    initModelDetailsModal();

    // 初始化图表
    initCharts();
});

/**
 * 初始化模型详情对话框
 */
function initModelDetailsModal() {
    // 获取所有"查看详细结构"按钮
    const viewModelButtons = document.querySelectorAll('.view-model-details');

    // 为每个按钮添加点击事件
    viewModelButtons.forEach(button => {
        button.addEventListener('click', function() {
            const modelType = this.getAttribute('data-model-type');
            openModelDetailsModal(modelType);
        });
    });
}

/**
 * 打开模型详情对话框
 * @param {string} modelType 模型类型 (mlp, lstm, cnn)
 */
function openModelDetailsModal(modelType) {
    // 获取模态框元素
    const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));

    // 显示加载状态
    document.getElementById('modelDetailsLoading').classList.remove('d-none');
    document.getElementById('modelDetailsContent').classList.add('d-none');
    document.getElementById('modelDetailsError').classList.add('d-none');

    // 更新模态框标题
    document.getElementById('modelDetailsModalLabel').textContent =
        `${modelType.toUpperCase()} 模型详细信息`;

    // 显示模态框
    modal.show();

    // 从API获取模型详情
    fetchModelDetails(modelType);
}

/**
 * 从API获取模型详情
 * @param {string} modelType 模型类型 (mlp, lstm, cnn)
 */
function fetchModelDetails(modelType) {
    fetch(`/viz/api/model-details/${modelType}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 显示模型详情
                displayModelDetails(data);
            } else {
                // 显示错误信息
                showModelDetailsError(data.message);
            }
        })
        .catch(error => {
            // 显示错误信息
            showModelDetailsError(`获取模型详情时出错: ${error}`);
        });
}

/**
 * 显示模型详情
 * @param {Object} data 模型详情数据
 */
function displayModelDetails(data) {
    // 隐藏加载状态，显示内容
    document.getElementById('modelDetailsLoading').classList.add('d-none');
    document.getElementById('modelDetailsContent').classList.remove('d-none');

    // 填充基本信息
    document.getElementById('modelName').textContent = data.model_name;
    document.getElementById('modelType').textContent = data.model_type;
    document.getElementById('totalParams').textContent = formatNumber(data.total_params);
    document.getElementById('trainableParams').textContent = formatNumber(data.trainable_params);
    document.getElementById('nonTrainableParams').textContent = formatNumber(data.non_trainable_params);
    document.getElementById('layersCount').textContent = data.layers_count;
    document.getElementById('inputShape').textContent = data.input_shape;
    document.getElementById('outputShape').textContent = data.output_shape;

    // 添加模型架构描述（如果存在）
    if (data.architecture_description) {
        // 检查是否已存在架构描述元素
        let architectureDescriptionElement = document.getElementById('architectureDescription');
        if (!architectureDescriptionElement) {
            // 创建架构描述元素
            const basicInfoTable = document.querySelector('#modelDetailsContent .row .col-md-6:first-child .card-body table');
            const architectureRow = document.createElement('tr');
            architectureRow.innerHTML = `
                <th style="width: 40%">架构描述</th>
                <td id="architectureDescription" class="text-muted small"></td>
            `;
            basicInfoTable.appendChild(architectureRow);
            architectureDescriptionElement = document.getElementById('architectureDescription');
        }

        // 设置架构描述内容
        architectureDescriptionElement.innerHTML = data.architecture_description.replace(/\n/g, '<br>').trim();
    }

    // 填充层表格
    const layersTable = document.getElementById('layersTable');
    layersTable.innerHTML = '';

    data.layers.forEach((layer, index) => {
        const row = document.createElement('tr');

        // 构建配置信息
        let configInfo = '';
        if (layer.units) configInfo += `单元数: ${layer.units}<br>`;
        if (layer.activation) configInfo += `激活函数: ${layer.activation}<br>`;
        if (layer.filters) configInfo += `过滤器数: ${layer.filters}<br>`;
        if (layer.kernel_size) configInfo += `核大小: ${layer.kernel_size}<br>`;
        if (layer.pool_size) configInfo += `池化大小: ${layer.pool_size}<br>`;
        if (layer.dropout_rate) configInfo += `丢弃率: ${layer.dropout_rate}<br>`;
        if (layer.strides) configInfo += `步长: ${layer.strides}<br>`;
        if (layer.padding) configInfo += `填充: ${layer.padding}<br>`;
        if (layer.regularizer) configInfo += `正则化: ${layer.regularizer}<br>`;
        if (layer.recurrent_dropout) configInfo += `循环丢弃率: ${layer.recurrent_dropout}<br>`;

        // 填充行内容
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${layer.name}</td>
            <td>${layer.type}</td>
            <td>${formatNumber(layer.params)}</td>
            <td>${layer.description || configInfo}</td>
            <td>${layer.input_shape}</td>
            <td>${layer.output_shape}</td>
        `;

        layersTable.appendChild(row);
    });

    // 更新图表
    updateModelMetricsChart(data.model_type.toLowerCase());
}

/**
 * 显示模型详情错误
 * @param {string} message 错误信息
 */
function showModelDetailsError(message) {
    document.getElementById('modelDetailsLoading').classList.add('d-none');
    document.getElementById('modelDetailsContent').classList.add('d-none');
    document.getElementById('modelDetailsError').classList.remove('d-none');
    document.getElementById('errorMessage').textContent = message;
}

/**
 * 初始化图表
 */
function initCharts() {
    // 准确率比较图表
    const accuracyCtx = document.getElementById('accuracyChart');
    if (accuracyCtx) {
        const modelMetrics = JSON.parse(document.getElementById('modelMetricsData').textContent);

        new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: ['MLP', 'LSTM', 'CNN'],
                datasets: [{
                    label: '准确率 (%)',
                    data: [
                        modelMetrics.mlp.accuracy,
                        modelMetrics.lstm.accuracy,
                        modelMetrics.cnn.accuracy
                    ],
                    backgroundColor: [
                        'rgba(13, 110, 253, 0.7)',
                        'rgba(25, 135, 84, 0.7)',
                        'rgba(13, 202, 240, 0.7)'
                    ],
                    borderColor: [
                        'rgb(13, 110, 253)',
                        'rgb(25, 135, 84)',
                        'rgb(13, 202, 240)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `准确率: ${context.raw.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 90,
                        max: 100
                    }
                }
            }
        });
    }

    // 误差指标比较图表
    const errorCtx = document.getElementById('errorChart');
    if (errorCtx) {
        const modelMetrics = JSON.parse(document.getElementById('modelMetricsData').textContent);

        new Chart(errorCtx, {
            type: 'bar',
            data: {
                labels: ['MLP', 'LSTM', 'CNN'],
                datasets: [{
                    label: 'RMSE',
                    data: [
                        modelMetrics.mlp.rmse,
                        modelMetrics.lstm.rmse,
                        modelMetrics.cnn.rmse
                    ],
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgb(255, 99, 132)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `RMSE: ${context.raw.toFixed(2)}`;
                            }
                        }
                    }
                }
            }
        });
    }
}

/**
 * 更新模型指标图表
 * @param {string} modelType 模型类型 (mlp, lstm, cnn)
 */
function updateModelMetricsChart(modelType) {
    const chartCtx = document.getElementById('modelMetricsChart');
    if (chartCtx) {
        const modelMetrics = JSON.parse(document.getElementById('modelMetricsData').textContent);
        const metrics = modelMetrics[modelType];

        // 清除现有图表
        if (chartCtx.chart) {
            chartCtx.chart.destroy();
        }

        // 创建新图表
        chartCtx.chart = new Chart(chartCtx, {
            type: 'radar',
            data: {
                labels: ['准确率', 'RMSE', 'MAE', 'R²'],
                datasets: [{
                    label: `${modelType.toUpperCase()} 模型性能`,
                    data: [
                        metrics.accuracy / 100, // 归一化准确率
                        1 - (metrics.rmse / 200), // 归一化RMSE (越低越好)
                        1 - (metrics.mae / 150),  // 归一化MAE (越低越好)
                        metrics.r2 || 0.9         // R² (已经是0-1范围)
                    ],
                    backgroundColor: getModelColor(modelType, 0.2),
                    borderColor: getModelColor(modelType, 1),
                    borderWidth: 2,
                    pointBackgroundColor: getModelColor(modelType, 1)
                }]
            },
            options: {
                scales: {
                    r: {
                        min: 0,
                        max: 1,
                        ticks: {
                            display: false
                        }
                    }
                }
            }
        });
    }
}

/**
 * 获取模型颜色
 * @param {string} modelType 模型类型 (mlp, lstm, cnn)
 * @param {number} alpha 透明度
 * @returns {string} 颜色值
 */
function getModelColor(modelType, alpha) {
    switch (modelType) {
        case 'mlp':
            return `rgba(13, 110, 253, ${alpha})`;
        case 'lstm':
            return `rgba(25, 135, 84, ${alpha})`;
        case 'cnn':
            return `rgba(13, 202, 240, ${alpha})`;
        default:
            return `rgba(108, 117, 125, ${alpha})`;
    }
}

/**
 * 格式化数字为千分位格式
 * @param {number} number 要格式化的数字
 * @returns {string} 格式化后的字符串
 */
function formatNumber(number) {
    return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
