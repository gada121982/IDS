import React from 'react'
import { Route, Switch } from 'react-router-dom'
// component
import AttackHistory from './views/AttackHistory'
import Monitor from './views/Monitor'

const routes = [
  {
    path: '/',
    exact: true,
    main: () => <Monitor />
  },
  {
    path: '/monitor',
    exact: true,
    main: () => <Monitor />
  },
  {
    path: '/attackhistory',
    exact: true,
    main: () => <AttackHistory />
  },
]

let activeRoute = () => {
  return (
    <Switch>
      {
        routes.map((route, index) => {
          return <Route component={route.main} path={route.path} exact={route.exact} key={index} />
        })
      }
    </Switch>
  )
}

export default activeRoute;