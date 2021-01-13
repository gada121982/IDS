import {Component} from 'react'
import socketClient from 'socket.io-client'

import './Monitor.scss'

const ENDPOINT = "http://localhost:8000"
let   COUNT = 0

class Monitor extends Component {

    constructor(props) {
        super(props)
        this.state = {
            queue: []
        }
    }

    componentDidMount() {
        const socket = socketClient.connect(ENDPOINT)
        let dashboard = document.querySelector('.monitor table')   
        let monitor = document.querySelector('.monitor')

        socket.on('traffic', (data) => {

            const trafficRow = document.querySelector('.monitor table tr:nth-child(2)')   
            let {features, result} = data

            COUNT = COUNT + 1
            const LENGTH = features ? features.length : 0
            const tr = document.createElement('tr')    
        
            const td_result = document.createElement('td')
            let result_content = document.createTextNode(result ? 'Danger' : 'OK')
            
            if(result === false){
                td_result.classList.add('false')
            } else {
                td_result.classList.add('true')
            }
            
            td_result.appendChild(result_content)
        
            
            const td_time = document.createElement('td')
            const time_content = document.createTextNode(new Date())
            td_time.appendChild(time_content)

            tr.appendChild(td_result)
            tr.appendChild(td_time)


            for (let i = 0; i < LENGTH; i++) {
                const td = document.createElement('td')
                const content = document.createTextNode(features[i])
                td.appendChild(content)
                tr.appendChild(td)
            }

            if (COUNT >= 20) {
                // remove table 0
                trafficRow.remove()
                dashboard.appendChild(tr)
                console.log('overload')

                monitor.scrollTop = monitor.scrollHeight - monitor.clientHeight;
            } else {
                // just append to
                dashboard.appendChild(tr)
                monitor.scrollTop = monitor.scrollHeight - monitor.clientHeight;
            }
        })
    }

    render(){
        return (
            <div className='monitor'>
                <table>
                    <tr>
                        <th>Detected</th>
                        <th>Time</th>
                        <th>dst_port</th>
                        <th>protocol</th>
                        <th>flow_duration</th>
                        <th>tot_fwd_pkts</th>
                        <th>tot_bwd_pkts</th>
                        <th>totlen_fwd_pkts</th>
                        <th>totlen_bwd_pkts</th>
                        <th>flow_byts/s</th>
                        <th>flow_pkts/s</th>
                        <th>fwd_iat_tot</th>
                        <th>bwd_iat_tot</th>
                        <th>fwd_psh_flags</th>
                        <th>bwd_psh_flags</th>
                        <th>fwd_urg_flags</th>
                        <th>bwd_urg_flags</th>
                        <th>fwd_header_len</th>
                        <th>bwd_header_len</th>
                        <th>fwd_pkts/s</th>
                        <th>bwd_pkts/s</th>
                        <th>pkt_len_var</th>
                        <th>fin_flag_cnt</th>
                        <th>syn_flag_cnt</th>
                        <th>rst_flag_cnt</th>
                        <th>psh_flag_cnt</th>
                        <th>ack_flag_cnt</th>
                        <th>urg_flag_cnt</th>
                        <th>cwe_flag_count</th>
                        <th>ece_flag_cnt</th>
                        <th>down/up_ratio</th>
                        <th>pkt_size_avg</th>
                        <th>fwd_seg_size_avg</th>
                        <th>bwd_seg_size_avg</th>
                        <th>bwd_blk_rate_avg</th>
                        <th>subflow_fwd_pkts</th>
                        <th>subflow_fwd_byts</th>
                        <th>subflow_bwd_pkts</th>
                        <th>subflow_bwd_byts</th>
                        <th>init_fwd_win_byts</th>
                        <th>init_bwd_win_byts</th>
                        <th>fwd_act_data_pkts</th>
                    </tr>
      
                </table>
            </div>
        )
    }
}

export default Monitor
