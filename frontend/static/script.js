// ============================================================
// ГЛОБАЛЬНОЕ СОСТОЯНИЕ
// ============================================================
let currentCacheKey = null;  // ключ обученной модели
let currentChart = null;

// ============================================================
// ПЕРЕКЛЮЧЕНИЕ ВКЛАДОК
// ============================================================
const tabs = document.querySelectorAll('.tab-btn');
const contents = document.querySelectorAll('.tab-content');
tabs.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        contents.forEach(c => c.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        tabs.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// ============================================================
// ТЕМА
// ============================================================
const themeToggle = document.getElementById('theme-toggle');
if (themeToggle) {
    const saved = localStorage.getItem('theme');
    if (saved === 'light') {
        document.body.classList.add('light-mode');
        themeToggle.innerText = '☀️';
    } else {
        themeToggle.innerText = '🌙';
    }
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('light-mode');
        const isLight = document.body.classList.contains('light-mode');
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
        themeToggle.innerText = isLight ? '☀️' : '🌙';
    });
}

// ============================================================
// ХЕЛПЕРЫ
// ============================================================
async function apiGet(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}

async function apiPost(url, body) {
    const r = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}

function setStatus(el, text, type = '') {
    el.innerText = text;
    el.className = 'status ' + type;
}

// ============================================================
// ВКЛАДКА 1: КАЛЬКУЛЯТОР
// ============================================================
const calcDataset = document.getElementById('calc-dataset');
const calcMethod = document.getElementById('calc-method');
const calcTrainBtn = document.getElementById('calc-train-btn');
const calcStatus = document.getElementById('calc-status');
const calcPredictBtn = document.getElementById('calc-predict-btn');

// Обновление видимых полей по датасету
function updateCalcFields() {
    const ds = calcDataset.value;
    document.getElementById('calc-latitude-row').style.display =
        (ds === '2d' || ds === '3d' || ds === '4d') ? 'flex' : 'none';
    document.getElementById('calc-season-row').style.display =
        (ds === '3d' || ds === '4d') ? 'flex' : 'none';
    document.getElementById('calc-w-row').style.display =
        (ds === '4d') ? 'flex' : 'none';

    // Обновляем hint по высоте
    apiGet(`/datasets/info/${ds}`).then(info => {
        const hMin = info.h_min || 0;
        const hMax = info.h_max || 20000;
        document.getElementById('calc-h-range').innerText =
            `(диапазон: ${hMin} — ${hMax} м)`;
        document.getElementById('calc-h').max = hMax;
    });
}
calcDataset.addEventListener('change', updateCalcFields);

// Обучение модели
calcTrainBtn.addEventListener('click', async () => {
    setStatus(calcStatus, 'Обучение...');
    try {
        const result = await apiPost('/ai/train', {
            dataset: calcDataset.value,
            method: calcMethod.value,
        });
        currentCacheKey = result.cache_key;
        setStatus(calcStatus, `✅ Модель обучена за ${result.train_time_s.toFixed(2)} с`, 'success');
    } catch (e) {
        setStatus(calcStatus, '❌ ' + e.message, 'error');
    }
});

// Предсказание
calcPredictBtn.addEventListener('click', async () => {
    if (!currentCacheKey) {
        alert('Сначала обучите модель');
        return;
    }
    try {
        const body = {
            cache_key: currentCacheKey,
            h: parseFloat(document.getElementById('calc-h').value),
        };
        const ds = calcDataset.value;
        if (ds === '2d' || ds === '3d' || ds === '4d') {
            body.latitude_deg = parseFloat(document.getElementById('calc-latitude').value);
        }
        if (ds === '3d' || ds === '4d') {
            body.season = document.getElementById('calc-season').value;
        }
        if (ds === '4d') {
            body.w = parseFloat(document.getElementById('calc-w').value);
        }

        const result = await apiPost('/ai/predict', body);
        document.getElementById('calc-T').innerText = result.output.T.toFixed(2);
        document.getElementById('calc-P').innerText = result.output.P.toFixed(0);
        document.getElementById('calc-rho').innerText = result.output.rho.toFixed(4);
        document.getElementById('calc-a').innerText = result.output.a.toFixed(2);
    } catch (e) {
        alert('Ошибка: ' + e.message);
    }
});

// ============================================================
// ВКЛАДКА 2: ИНТЕРВАЛ
// ============================================================
const rangeCalcBtn = document.getElementById('range-calc-btn');
const rangeExportBtn = document.getElementById('range-export-btn');
const rangeTbody = document.getElementById('range-tbody');
let rangeData = [];

rangeCalcBtn.addEventListener('click', async () => {
    if (!currentCacheKey) {
        alert('Сначала обучите модель на вкладке «Калькулятор»');
        return;
    }
    try {
        const body = {
            cache_key: currentCacheKey,
            h_start: parseFloat(document.getElementById('range-h-start').value),
            h_end: parseFloat(document.getElementById('range-h-end').value),
            h_step: parseFloat(document.getElementById('range-h-step').value),
            latitude_deg: parseFloat(document.getElementById('range-latitude').value),
            season: document.getElementById('range-season').value,
            w: parseFloat(document.getElementById('range-w').value),
        };
        const result = await apiPost('/ai/predict_range', body);
        rangeData = result.data;
        rangeTbody.innerHTML = '';
        result.data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.h}</td>
                <td>${row.T.toFixed(2)}</td>
                <td>${row.P.toFixed(0)}</td>
                <td>${row.rho.toFixed(4)}</td>
                <td>${row.a.toFixed(2)}</td>
            `;
            rangeTbody.appendChild(tr);
        });
    } catch (e) {
        alert('Ошибка: ' + e.message);
    }
});

rangeExportBtn.addEventListener('click', () => {
    if (rangeData.length === 0) {
        alert('Сначала рассчитайте интервал');
        return;
    }
    let csv = 'h,T,P,rho,a\n';
    rangeData.forEach(r => {
        csv += `${r.h},${r.T},${r.P},${r.rho},${r.a}\n`;
    });
    const blob = new Blob([csv], {type: 'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'msa_range.csv';
    a.click();
});

// ============================================================
// ВКЛАДКА 3: ГРАФИКИ
// ============================================================
const chartBuildBtn = document.getElementById('chart-build-btn');

chartBuildBtn.addEventListener('click', async () => {
    const ds = document.getElementById('chart-dataset').value;
    const param = document.getElementById('chart-param').value;
    const hMin = parseFloat(document.getElementById('chart-h-min').value);
    const hMax = parseFloat(document.getElementById('chart-h-max').value);
    const latitude = parseFloat(document.getElementById('chart-latitude').value);
    const season = document.getElementById('chart-season').value;

    try {
        const url = `/reference/${ds}?param=${param}&h_min=${hMin}&h_max=${hMax}&latitude_deg=${latitude}&season=${season}`;
        const result = await apiGet(url);

        const heights = result.data.map(d => d.h);
        const values = result.data.map(d => d.value);

        const ctx = document.getElementById('chart-canvas').getContext('2d');
        if (currentChart) currentChart.destroy();
        currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: heights,
                datasets: [{
                    label: `${param} — ${result.standard}`,
                    data: values,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { labels: { color: document.body.classList.contains('light-mode') ? '#0f172a' : '#e2e8f0' } }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Высота, м', color: document.body.classList.contains('light-mode') ? '#0f172a' : '#e2e8f0' },
                        ticks: { color: document.body.classList.contains('light-mode') ? '#0f172a' : '#94a3b8' }
                    },
                    y: {
                        title: { display: true, text: param, color: document.body.classList.contains('light-mode') ? '#0f172a' : '#e2e8f0' },
                        ticks: { color: document.body.classList.contains('light-mode') ? '#0f172a' : '#94a3b8' }
                    }
                }
            }
        });
    } catch (e) {
        alert('Ошибка: ' + e.message);
    }
});

// ============================================================
// ВКЛАДКА 4: ОБУЧЕНИЕ
// ============================================================
const trainDataset = document.getElementById('train-dataset');
const trainMethod = document.getElementById('train-method');
const trainDefaultBtn = document.getElementById('train-default-btn');
const trainBtn = document.getElementById('train-btn');
const trainStatus = document.getElementById('train-status');
const hyperparamsPanel = document.getElementById('hyperparams-panel');
let currentHyperparams = {};

async function loadDefaultHyperparams() {
    try {
        const params = await apiGet(`/ai/default_hyperparams/${trainMethod.value}`);
        currentHyperparams = params;
        renderHyperparamsPanel(params);
    } catch (e) {
        hyperparamsPanel.innerHTML = '<p>Ошибка загрузки гиперпараметров</p>';
    }
}

function renderHyperparamsPanel(params) {
    hyperparamsPanel.innerHTML = '<h4>Гиперпараметры:</h4>';
    for (const [key, value] of Object.entries(params)) {
        const row = document.createElement('div');
        row.className = 'form-row';
        row.innerHTML = `
            <label>${key}:</label>
            <input type="text" data-param="${key}" value='${JSON.stringify(value)}'>
        `;
        hyperparamsPanel.appendChild(row);
    }
}

trainDefaultBtn.addEventListener('click', loadDefaultHyperparams);
trainMethod.addEventListener('change', loadDefaultHyperparams);

trainBtn.addEventListener('click', async () => {
    setStatus(trainStatus, 'Обучение...');
    try {
        // Собираем гиперпараметры из полей
        const params = {};
        hyperparamsPanel.querySelectorAll('input[data-param]').forEach(inp => {
            try {
                params[inp.dataset.param] = JSON.parse(inp.value);
            } catch {
                params[inp.dataset.param] = inp.value;
            }
        });

        const result = await apiPost('/ai/train', {
            dataset: trainDataset.value,
            method: trainMethod.value,
            hyperparams: params,
            test_size: parseFloat(document.getElementById('train-test-size').value),
        });

        currentCacheKey = result.cache_key;
        document.getElementById('train-result').style.display = 'block';
        document.getElementById('train-cache-key').innerText = result.cache_key;
        document.getElementById('train-time').innerText = result.train_time_s.toFixed(2);
        document.getElementById('train-mae').innerText = result.metrics.MAE_avg.toFixed(4);
        document.getElementById('train-r2').innerText = result.metrics.R2_avg.toFixed(4);

        setStatus(trainStatus, '✅ Модель обучена', 'success');
        await refreshValidationList();
    } catch (e) {
        setStatus(trainStatus, '❌ ' + e.message, 'error');
    }
});

// ============================================================
// ВКЛАДКА 5: ВАЛИДАЦИЯ
// ============================================================
const validCacheKey = document.getElementById('valid-cache-key');
const validLoadBtn = document.getElementById('valid-load-btn');

async function refreshValidationList() {
    try {
        const list = await apiGet('/ai/list_trained');
        validCacheKey.innerHTML = '<option value="">— выберите модель —</option>';
        list.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item.cache_key;
            opt.innerText = `${item.dataset} / ${item.method} (R²=${item.metrics_avg?.toFixed(4)})`;
            validCacheKey.appendChild(opt);
        });
        if (currentCacheKey) validCacheKey.value = currentCacheKey;
    } catch (e) {
        console.error(e);
    }
}

validLoadBtn.addEventListener('click', async () => {
    const key = validCacheKey.value;
    if (!key) {
        alert('Выберите модель');
        return;
    }
    try {
        const r = await apiGet(`/verify/${encodeURIComponent(key)}`);
        document.getElementById('valid-result').style.display = 'block';
        document.getElementById('valid-dataset').innerText = r.dataset;
        document.getElementById('valid-method').innerText = r.method;
        document.getElementById('valid-standard').innerText = 'ГОСТ Р 70469-2026 / ISO 5878:2026';

        const tbody = document.getElementById('valid-tbody');
        tbody.innerHTML = '';
        r.metrics_by_param.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.parameter}</td>
                <td>${row.mae.toFixed(4)}</td>
                <td>${row.rmse.toFixed(4)}</td>
                <td>${row.r2.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        alert('Ошибка: ' + e.message);
    }
});

// =================================