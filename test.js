const fetch = require('node-fetch');


// fetch("http://111.229.108.100:8010/embedding_doc", {
//   method: "POST",
//   body: JSON.stringify({
//     urls: '123'
//   }),
//   headers: {
//     "Content-type": "application/json; charset=UTF-8"
//   }
// })
//   .then((response) => response.json())
//   .then((json) => console.log(json));

// fetch('http://www.baidu.com')
// .then((response) => {
//     console.log(response); 
// }).catch((error) => {
//     console.log(`Error: ${error}`);
// })

// console.log(typeof 1)


// 当前运行结果: (object)

// {
//   "cloud://lowcode-7glmipr1e4e8cd77.6c6f-lowcode-7glmipr1e4e8cd77-1316499535/weda-uploader/cf5b401b9083d45186889266958ab1e3-test.js": "https://6c6f-lowcode-7glmipr1e4e8cd77-1316499535.tcb.qcloud.la/weda-uploader/cf5b401b9083d45186889266958ab1e3-test.js?sign=eb82b1d6856b066ad5b4f2f7b04506c4&t=1710178413",
//   "cloud://lowcode-7glmipr1e4e8cd77.6c6f-lowcode-7glmipr1e4e8cd77-1316499535/weda-uploader/b1d026a3f06182087b54caafb766d4f3-index.html": "https://6c6f-lowcode-7glmipr1e4e8cd77-1316499535.tcb.qcloud.la/weda-uploader/b1d026a3f06182087b54caafb766d4f3-index.html?sign=914ce28e117bdaaa46712c4f2486dc74&t=1710178413",
//   "cloud://lowcode-7glmipr1e4e8cd77.6c6f-lowcode-7glmipr1e4e8cd77-1316499535/weda-uploader/3f9f4a03512854824857508141549981-index.html": "https://6c6f-lowcode-7glmipr1e4e8cd77-1316499535.tcb.qcloud.la/weda-uploader/3f9f4a03512854824857508141549981-index.html?sign=63b0087384ece8c106c0b6206b2c1015&t=1710178413",
//   "cloud://lowcode-7glmipr1e4e8cd77.6c6f-lowcode-7glmipr1e4e8cd77-1316499535/weda-uploader/2a7a064d3fd2babcaa3b2f7fb762127e-架构.md": "https://6c6f-lowcode-7glmipr1e4e8cd77-1316499535.tcb.qcloud.la/weda-uploader/2a7a064d3fd2babcaa3b2f7fb762127e-架构.md?sign=bb4aa3b1040548f15275000606c2c2db&t=1710178413"
// }




let p = {
method: 'POST',
headers: {
'Content-Type': 'application/json',
},
body: JSON.stringify({"123":"456"}),
};

const response = await fetch('http://111.229.108.100:8010/embedding_doc',p);

const result = await response.json();

// 这里返回数据，和出参结构映射
return result;
  
