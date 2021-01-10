const axios = require('axios');


let predict = async  (feature) => {
    let data = await axios.post('http://localhost:5000/predict', {feature}).then(res => res.data)
    
    console.log('this is data', data)
    return data
}


module.exports = {predict}