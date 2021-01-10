const express = require('express');
const bodyParser = require('body-parser');
const {predict} = require('./lib/predict')
const cors = require('cors')

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
  let result = await predict(features)

  console.log(features.length)
  console.log('socket length', SOCKET.length)
  SOCKET.forEach((socket) => {
    socket.emit('traffic', {result, features})
  }) 
  res.status(200).send({result})
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

   