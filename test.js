const axios = require('axios');
const fs =  require('fs')



let features = [443,	6,	141385,	9,	7,	553,	3773.0,	30597.30523,
113.166177,	141385.0,	51417.0,	0,	0,	0,	0,	192,
152,	63.655975,	49.510203,	225352.389700,	0,	0,
1,	1,	0,	0,	0,	1,	0,	270.375000,	61.444444,	539.000000,
0,	9,	553,	7,	3773,	8192,	119,	4]

let featuresAttack = [8080,6,16079,3,4,326,129.0,28297.77971,
435.350457,645.0,15513.0,0,0,0,0,72,92,
186.578767,248.771690,13318.69643,0,0,1,1,0,0,0,
1,1,65.0,108.666667,32.25,0,3,326,4,129,8192,219,1]

console.log(features.length, featuresAttack.length)


let sendTraffic = async () => {

    features[0] = Math.floor(Math.random()*500) + 1
    let result = {}

    if(features[0] % 2 === 0) {
        result = await axios.post('http://localhost:8000/feature',{ features: features}).then(res => res.data)
    } else {
        result = await axios.post('http://localhost:8000/feature',{ features: featuresAttack}).then(res => res.data)
    }

    console.log(result)
}

JSON.parse

setInterval(()=> {
    sendTraffic()
}, 1000)