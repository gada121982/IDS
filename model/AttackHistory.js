const { Schema, model } = require('mongoose')

const AttackHistory = new Schema({
  features: {
    type: Array,
    required: true
  },
  result: {
    type: Boolean,
    required: true
  }
}, {
  timestamps: true
})

module.exports = model('AttackHistory', AttackHistory, 'AttackHistory')