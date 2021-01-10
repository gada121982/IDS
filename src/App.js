import {Component} from 'react'
import logoUit from './images/logo-uit.png'
import './App.scss';

import activeRoute from './router'
import { NavLink } from 'react-router-dom'

import { BrowserRouter as Router } from 'react-router-dom'

class App extends Component{

  constructor(props) {
    super(props)

    this.state = {
      time: '',
      date: '',
      active: 1
    }
  }

  componentDidMount() {
    setInterval(() => {
      this.setState({
        time: `${new Date().getHours()} : ${new Date().getMinutes()} : ${new Date().getSeconds()}`,
        date: `Ngày ${new Date().getDate()}, Tháng ${new Date().getMonth() + 1}, ${new Date().getFullYear()}`
      })

    }, 1000)
  }

  onRouting = () => {
    let state = this.state.active === 1 ? 2 : 1
    this.setState({
      active: state
    })
  }

  render(){
    let {time, date, active } = this.state
    return(
      <div className="wrapper">
        <header>
          <img src={logoUit}/>
          <div className="date-display">
            <span>{time}</span>
            <span>{ date}</span>
          </div>
          <h3>Hệ thống IDS</h3>
        </header>
        
        <Router>
        <div className="control">
          <NavLink exact to="/monitor"
            className={active === 1 ? "to-monitor active" : "to-monitor"}
            onClick={this.onRouting}>
            Monitor
          </NavLink>

          <NavLink exact to="/attackhistory"
            className={active === 2 ? "to-monitor active" : "to-monitor"}
            onClick={this.onRouting}>
            Attack history
          </NavLink> 
        </div>

        <div className="main">
        {
          activeRoute()
        }
        </div>
        </Router>

      </div>
    )
    
  }
}

export default App;
