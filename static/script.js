let selectedPeriod = "1";

// Period buttons
function loadData(period) {
    selectedPeriod = period;
    document.querySelectorAll('.period-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.period-btn:nth-child(${period})`).classList.add('active');

    $.get(`/get_data/${period}`, function (data) {
        const tableBody = document.getElementById('table-body');
        tableBody.innerHTML = '';
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${row.No}</td><td>${row.Name}</td><td>${row["Register Number"]}</td>`;
            tableBody.appendChild(tr);
        });
    });
}

// Start recognition
$('#start-btn').on('click', function () {
    fetch("/start_recognition", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ period: selectedPeriod })
    })
        .then(res => res.json())
        .then(data => alert(data.message));
});

// Stop & save
$('#stop-btn').on('click', function () {
    $.post('/create_csv', {}, function (response) {
        alert(response.message);
    });
});
