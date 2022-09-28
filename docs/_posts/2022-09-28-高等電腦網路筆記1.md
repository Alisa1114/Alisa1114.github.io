---
layout: post  
title:  "高等電腦網路Review Question."  
date:   2022-09-20 11:12:53  
tags: Computer_Networks
math: true
---  

### 參考資料：Computer Networking: A Top-Down Approach，James F. Kurose and Keith W. Ross

## 1. List the basic components of Internet stack.
> * network edge
> * access networks, physical media
> * network core


## 2. List the types of access networks. At least five types.  
> Digital Subscriber Line (DSL), cable modems, Fiber to the home, Ethernet internet access, Wireless access networks.


## 3. What is Protocol?
> Protocols define format, order of messages sent and received among network entities, and actions taken on message transmission, receipt


## 4. What is packet switching?
> Each end-end data stream divided into packets
>
> User A, B packets share network resources
>
> Each packet uses full link bandwidth
>
> Resources used as needed
> 
> Resource contention: aggregate resource demand can exceed amount available
> 
> Congestion: packets queue, wait for link use
> 
> Store and forward: packets move one hop at a time


## 5. What is Nodal Delay Mode?
> Nodal Delay is the summation of processing delay, queuing delay, transmission delay, propagation delay
>
> $$ d_{nodal} = d_{porc} + d_{queue} + d_{trans} + d_{prop} $$


## 6. What is the delay that can be very large, and why it is?
> Queuing delay can be very large
> 
> If its traffic intensity $La/R$ is larger than 1, average queuing delay can reach infinite
> 
> When $La/R < 1$, queuing delay is calculated as $I(L/R)(1 - I)$ and $I = La/R$
>
> 所以 average queuing delay 呈平方成長


## 7. What are the five layers of Internet Protocol stack and their functionalities?
> application: supporting network applications
> * FTP, SMTP, HTTP
>
> transport: process-process data transfer
> * TCP, UDP
> 
> network: routing of datagrams from source to destination
> * IP, routing protocols
> 
> link: data transfer between neighboring network elements
> * PPP, Ethernet
> 
> physical: bit "on the wire"


## 8. What is the characteristics of layering design in Internet Protocol stack?
> Explicit structure allows identification, relationship of complex system's pieces
> 
> Modularization eases maintenance, updating of system


## 9.  How does encapsulation work in internet protocol stack?
> message會從application layer開始往下傳，到transport layer, network layer, link layer時會給message加上header information

## 10. What are the layers that the hub, switch and router handle the packets?
> network, link, physical