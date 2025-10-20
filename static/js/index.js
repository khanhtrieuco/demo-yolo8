

async function submit() {
    let image_url = document.getElementById('image_url').value || "";
    let description = document.getElementById('description').value || "";
    let code = document.getElementById('code').value || "";
    const formData = new FormData();
    const uploadBtn = document.getElementById('uploadBtn');
    const spinner = document.getElementById('spinner');
    formData.append("image_url", image_url);
    formData.append("description", description);
    formData.append("code", code);
    try {
        uploadBtn.style.display = 'none';
        spinner.style.display = 'block';
        const res = await axios.post("/upload/reference", formData, {
            headers: { "Content-Type": "multipart/form-data" }
        });
        if (parseInt(res.data.code) == 400) {
            const alert = document.getElementById('alert-fail');

            let html = document.querySelector('#alert-fail > strong');
            html.innerHTML = `${res.data.message}`;
            alert.style.display = 'block';
            setTimeout(() => {
                alert.style.display = 'none';
            }, 2000);

            console.log('btnClose', closeBtn);

        }
        if (parseInt(res.data.code) == 200) {
            const alert = document.getElementById('alert-success');
            const resultBlock = document.getElementById('result');
            // const closeBtn = document.querySelector('.modal-footer > button');
            resultBlock.innerHTML = `
                <p>Kết quả upload mẫu</p>
                <p>Description : ${res.data.data.description}</p>
                <p>Path : ${res.data.data.path}</p>
                <p>reference_id : ${res.data.data.reference_id}</p>
            `
            resultBlock.style.display = 'block'
            alert.style.display = 'block';
            setTimeout(() => {
                alert.style.display = 'none';
                // closeBtn.click();
                location.reload();
            }, 2000);
            document.getElementById('image').remove();
            document.getElementById('image_url').value = "";
            document.getElementById('description').value = "";
            document.getElementById('code').value = "";

        }
        console.log('response: ', res.data);
    } catch (error) {
        console.log('Error', error);
    } finally {
        uploadBtn.style.display = 'block';
        spinner.style.display = 'none';
    }
}
async function handleOnchange() {
    let image_url = document.getElementById('image_url').value || "";
    let image = document.getElementById('image');
    if (image_url !== "") {

        image.setAttribute("src", `${image_url}`)
        image.setAttribute("width", "100")
        image.setAttribute("height", "100")
        image.setAttribute("class", "mt-1")
    } else if (image_url === "") {

        image.setAttribute("src", "")
    }
}
async function check() {
    let image_url_check = document.getElementById('image_url_check').value || "";
    let reference_code = document.getElementById('code_check').value || "";
    const formData = new FormData();
    const checkBtn = document.getElementById('checkBtn');
    const spinner = document.getElementById('spinnerCheck');
    formData.append("image_url_check", image_url_check);
    formData.append("reference_code", reference_code);
    try {
        checkBtn.style.display = 'none';
        spinner.style.display = 'block';
        const res = await axios.post("/upload/test", formData, {
            headers: { "Content-Type": "multipart/form-data" }
        });
        if (parseInt(res.data.code) == 200) {
            const alert = document.getElementById('alert-success');
            const resultBlock = document.getElementById('checkResult');
            // const closeBtn = document.querySelector('.modal-footer > button');
            resultBlock.innerHTML = `
                <p>Kết quả check mẫu</p>
                <p>Summary : ${res.data.data.summary}</p>
                <p>Status : ${res.data.data.status}</p>
            `
            resultBlock.style.display = 'block';
            alert.style.display = 'block';
            checkBtn.style.display = 'none';
            document.getElementById('image_url_check').remove();
            document.getElementById('image_url_check').value = "";
            document.getElementById('code_check').value = "";

        }
        console.log('response: ', res.data);
    } catch (error) {
        console.log("Err", error);
    } finally {
        checkBtn.style.display = 'block';
        spinner.style.display = 'none';
    }
}
async function handleOnchangeCheck() {
    let image_url = document.getElementById('image_url_check').value || "";
    let image = document.getElementById('image_check');
    if (image_url !== "") {
        image.setAttribute("src", `${image_url}`)
        image.setAttribute("width", "100")
        image.setAttribute("height", "100")
        image.setAttribute("class", "mt-1")
    } else if (image_url === "") {
        image.setAttribute("src", "");
        image.setAttribute("width", "0")
        image.setAttribute("height", "0")
    }
}

async function viewHistory() {
    const historyButtons = document.querySelectorAll('.btn-history');
    const tableBody = document.querySelector('#checkRefHistoryModal tbody');
    historyButtons.forEach(btn => {
        btn.addEventListener('click', async function () {
            const refId = this.getAttribute('data-ref-id');
            console.log('Xem lịch sử của Reference_id:', refId);
            // Xóa nội dung cũ
            tableBody.innerHTML = '<tr><td colspan="6">Đang tải...</td></tr>';

            try {
                const res = await axios.get(`/check-history/${refId}`)
                
                if (!res.data.data || res.data.data.length === 0) {
                    tableBody.innerHTML = '<tr><td colspan="6">Không có dữ liệu lịch sử.</td></tr>';
                    return;
                }
                console.log("data: ", res.data);
                // Render dữ liệu
                tableBody.innerHTML = res.data.data.map(item => `
                    <tr>
                        <td>${item.reference_id}</td>
                        <td><img src="${item.test_path}" width="100" height="100"/></td>
                        <td>${item.similarity}%</td>
                        <td>${item.result}</td>
                        <td style="word-wrap: break-word; word-break: break-word; white-space: pre-wrap; width: 400px;">${item.issues || ''}</td>
                        <td>${new Date(item.checked_at).toLocaleString()}</td>
                    </tr>
                    `).join('');
            } catch (err) {
                console.error('Lỗi khi tải lịch sử:', err);
                tableBody.innerHTML = '<tr><td colspan="6" class="text-danger">Lỗi khi tải dữ liệu.</td></tr>';
            }
        });
    });
}