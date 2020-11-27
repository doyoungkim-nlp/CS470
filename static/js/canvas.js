let isDrawing = false;
let x = 0;
let y = 0;

const myPics = document.getElementById("canvas");
const saveBtn = document.getElementById("btn")
const context = myPics.getContext('2d');

// event.offsetX, event.offsetY gives the (x,y) offset from the edge of the canvas.

// Add the event listeners for mousedown, mousemove, and mouseup
myPics.addEventListener('mousedown', e => {
  x = e.offsetX;
  y = e.offsetY;
  isDrawing = true;
});

myPics.addEventListener('mousemove', e => {
  if (isDrawing === true) {
    drawLine(context, x, y, e.offsetX, e.offsetY);
    x = e.offsetX;
    y = e.offsetY;
  }
});

window.addEventListener('mouseup', e => {
  if (isDrawing === true) {
    drawLine(context, x, y, e.offsetX, e.offsetY);
    x = 0;
    y = 0;
    isDrawing = false;
  }
});

saveBtn.addEventListener('click', handleSave)

function drawLine(context, x1, y1, x2, y2) {
  context.beginPath();
  context.strokeStyle = 'white';
  context.lineWidth = 1;
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.closePath();
}


function handleSave() {
  const image = canvas.toDataURL("image/jpeg", 1.0);
  var currentdate = new Date(); 

  var year    = currentdate.getFullYear();
  var month   = currentdate.getMonth() + 1; 
  var date    = currentdate.getDate();
  var hour    = currentdate.getHours();
  var minute  = currentdate.getMinutes();
  var second  = currentdate.getSeconds(); 

  if(month.toString().length == 1) {
      month = '0' + month;
  }
  if(date.toString().length == 1) {
      date = '0' + date;
  }   
  if(hour.toString().length == 1) {
      hour = '0' + hour;
  }
  if(minute.toString().length == 1) {
      minute = '0' + minute;
  }
  if(second.toString().length == 1) {
      second = '0' + second;
  }   

  var dateTime = year + "." + month  + "." 
                  + date + "." + hour + ":"  
                  + minute + ":" + second;

console.log("about to send")
  $.ajax({
    type: "POST",
    url: "http://222.114.173.63:5000/result",
    data:{
      dateTime : dateTime,
      imageBase64: image
    }
  }).done(function() {
    console.log('sent');
  });
}
