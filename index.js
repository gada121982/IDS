const express = require('express');
const bodyParser = require('body-parser');
const {predict} = require('./lib/predict')
const cors = require('cors')
const db = require('mongoose')
const AttackHistory = require('./model/AttackHistory')

console.log(AttackHistory)

const DBURL = `mongodb+srv://doan:doan123456@cluster0.py1f7.mongodb.net/doan?retryWrites=true&w=majority`

db.connect(DBURL, {useNewUrlParser: true, useUnifiedTopology: true});


app = express()
const server = require('http').Server(app)
const io = require('socket.io')(server)

app.use(cors())
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())



app.set('view engine', 'ejs')
app.set('views', './views')
app.use(bodyParser.json())
app.use(
  bodyParser.urlencoded({
    extended: true
  })
)

app.use(express.static('public'))

let SOCKET = []


app.get('/', (req, res) => {
    res.render('index')
})

app.post('/feature', async (req, res) => {
  let {features} = req.body
  features = features.replace('[', ' ').replace(']', ' ').trim().split(',')

  let features_clean = []

  for(let i = 0; i < features.length ; i++) {
    features_clean.push(parseFloat(features[i]))
  }


  console.log('feature', features_clean)

  let result = ''
  try {
    result = await predict(features_clean)
  } catch (err) {
    console.log('error predict', err)
  }
  
  SOCKET.forEach((socket) => {
    socket.emit('traffic', {result, features})
  }) 
  if(result) {
    // save to db
    const attack = new AttackHistory({
      features,
      result
    })
    await attack.save()
    
    console.log('on danger')
    SOCKET.forEach((socket) => {
      socket.emit('danger', {result, features})
    }) 
      
  }
  res.status(200).send({result})
 

})

app.get('/attackhistory', async (req, res) => {
  let data = await AttackHistory.find({}).limit(50)
  console.log('data', data)
  res.send(data)

})

io.on('connection', function(socket){
  // console.log('socket')
  SOCKET.push(socket)

  console.log("socket", socket.id)
  socket.on('disconnect', function(){
    console.log('disconnected')
  })
})


server.listen(8000, function() {
    console.log('app listen on port 8000')
})

   