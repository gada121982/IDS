const axios = require('axios');

let sendTraffic = async () => {
    let result = await axios.post('http://localhost:8000/feature', {
        features: [ 443, 6, '02/03/2018 08:47', 141385, 9, 7,
        553, 3773, 30597.30523, 113.166177, 141385,
        51417, 0, 0, 0, 0, 192, 152, 63.655975, 49.510203,
        225352.3897, 0, 0, 1, 1, 0, 0, 0, 1, 0, 270.375,
        61.444444, 539, 0, 0, 0, 0, 0, 0, 9, 553, 7, 3773, 8192, 119, 4]
    }).then(res => res.data)

    console.log(result)
}

setInterval(()=> {
    sendTraffic()
}, 1000)