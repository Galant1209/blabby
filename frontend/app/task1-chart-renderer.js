/*
 * Shared Task 1 chart renderer.
 *
 * writing.html and admin.html both call this exact entry point. Keeping the
 * deterministic pie renderer here makes the admin human-review preview the
 * same renderer a student sees; it is not a second approximation.
 */
(function (root) {
    const PIE_COLOURS = ['#1A3550', '#C9A84C', '#2D5016', '#6B1A1A', '#7A4B7A', '#356A78', '#8A5A2B', '#56636F'];

    // PURE_GEOMETRY_START
    function calculatePieSlices(values, cx, cy, radius) {
        const total = values.reduce((sum, value) => sum + value, 0);
        let cursor = -Math.PI / 2;
        return values.map((value, index) => {
            const startAngle = cursor;
            const endAngle = index === values.length - 1
                ? -Math.PI / 2 + Math.PI * 2
                : startAngle + (value / total) * Math.PI * 2;
            cursor = endAngle;
            return { index, value, cx, cy, radius, offset: 0, startAngle, endAngle };
        });
    }
    // PURE_GEOMETRY_END

    function validatePieChartData(raw) {
        if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
        const allowed = new Set(['chart_type', 'title', 'labels', 'values', 'unit']);
        if (Object.keys(raw).some(key => !allowed.has(key))) return null;
        if (raw.chart_type !== 'pie_chart' || typeof raw.title !== 'string' || typeof raw.unit !== 'string') return null;
        if (!Array.isArray(raw.labels) || !Array.isArray(raw.values)) return null;
        if (raw.labels.length < 2 || raw.labels.length > 8 || raw.labels.length !== raw.values.length) return null;
        if (raw.labels.some(label => typeof label !== 'string' || !label.trim() || label.length > 80)) return null;
        if (raw.values.some(value => typeof value !== 'number' || !Number.isFinite(value) || value < 0)) return null;
        const total = raw.values.reduce((sum, value) => sum + value, 0);
        if (!(total > 0)) return null;
        if (['%', 'percent', 'percentage'].includes(raw.unit.trim().toLowerCase()) && (total < 99 || total > 101)) return null;
        return {
            chart_type: 'pie_chart',
            title: raw.title.replace(/\s+/g, ' ').trim(),
            labels: raw.labels.map(label => label.replace(/\s+/g, ' ').trim()),
            values: raw.values.slice(),
            unit: raw.unit.replace(/\s+/g, ' ').trim(),
        };
    }

    function renderPieChart(raw, container) {
        const data = validatePieChartData(raw);
        if (!container) return false;
        container.replaceChildren();
        if (!data) return false;

        const logicalWidth = Math.max(240, Math.min(container.clientWidth || 520, 520));
        const logicalHeight = 280;
        const dpr = Math.max(1, root.devicePixelRatio || 1);
        const canvas = document.createElement('canvas');
        canvas.className = 'pie-chart-canvas';
        canvas.width = Math.round(logicalWidth * dpr);
        canvas.height = Math.round(logicalHeight * dpr);
        canvas.style.aspectRatio = `${logicalWidth} / ${logicalHeight}`;
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', data.title);
        container.appendChild(canvas);

        const context = canvas.getContext('2d');
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        context.clearRect(0, 0, logicalWidth, logicalHeight);
        const cx = logicalWidth / 2;
        const cy = logicalHeight / 2;
        const radius = Math.min(110, logicalWidth * 0.34, logicalHeight * 0.39);
        const slices = calculatePieSlices(data.values, cx, cy, radius);
        slices.forEach(slice => {
            context.beginPath();
            context.moveTo(slice.cx, slice.cy);
            context.arc(slice.cx, slice.cy, slice.radius, slice.startAngle, slice.endAngle);
            context.closePath();
            context.fillStyle = PIE_COLOURS[slice.index % PIE_COLOURS.length];
            context.fill();
            context.strokeStyle = '#ffffff';
            context.lineWidth = 1;
            context.stroke();
        });

        const legend = document.createElement('div');
        legend.className = 'pie-chart-legend';
        data.labels.forEach((label, index) => {
            const item = document.createElement('div');
            item.className = 'pie-chart-legend-item';
            const swatch = document.createElement('span');
            swatch.className = 'pie-chart-swatch';
            swatch.style.backgroundColor = PIE_COLOURS[index % PIE_COLOURS.length];
            const text = document.createElement('span');
            text.className = 'pie-chart-label';
            text.textContent = `${label}: ${data.values[index]}` + (data.unit === '%' ? '%' : ` ${data.unit}`);
            item.append(swatch, text);
            legend.appendChild(item);
        });
        container.appendChild(legend);
        return true;
    }

    function renderLegacyChartImage(svg, container) {
        if (!container) return false;
        container.replaceChildren();
        if (!svg) return false;
        const image = document.createElement('img');
        image.alt = 'IELTS Task 1 chart';
        image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(String(svg))}`;
        image.style.width = '100%';
        image.style.height = 'auto';
        container.appendChild(image);
        return true;
    }

    root.BlabbyTask1ChartRenderer = Object.freeze({
        calculatePieSlices,
        validatePieChartData,
        renderPieChart,
        renderLegacyChartImage,
    });
})(typeof window !== 'undefined' ? window : globalThis);
