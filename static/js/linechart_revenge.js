google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawCurveTypes);
var revLinelist = [];
var revright = 0;
var revlen = 0;
$.getJSON('../static/history.json', function(data) {         
  $.each( data.history, function( key, val ){
    console.log(val.category);
    if(val.category=="revenge"){
      if(val.correctness=="revright!")
      {
        revright +=1;
      }
      revlen +=1;
      revLinelist.push([revlen, revright/revlen]);
      console.log(revlen, revright);
      console.log(revLinelist);
    }
  });
});
function drawCurveTypes() {

      var data = new google.visualization.DataTable();

      data.addColumn('number', 'X');
      data.addColumn('number', 'revenge');
      
      data.addRows(revLinelist);
      console.log(revLinelist);


      var options = {
        hAxis: {
          title: 'Attmepts'
        },
        vAxis: {
          title: 'Accuracy'
        },
        series: {
          1: {curveType: 'function'}
        }
      };

      var chart = new google.visualization.LineChart(document.getElementById('linechart_revenge_div'));
      chart.draw(data, options);
    }