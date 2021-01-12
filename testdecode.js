

let features = '[1,2,3,4,5]'

features = features.replace('[', ' ').replace(']', ' ').trim().split(',')

let result = []

for(let i = 0; i < features.length ; i++) {
  result.push(parseFloat(features[i]))
}

console.log(result)