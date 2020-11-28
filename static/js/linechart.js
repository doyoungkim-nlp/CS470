google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawCurveTypes);
var linelist = [];
var right = 0;
var len = 0;
$.getJSON('../static/history.json', function(data) {         
  $.each( data.history, function( key, val ){
    console.log(val.correctness);
    if(val.correctness=="right!")
    {
      right +=1;
    }
    len +=1;
    linelist.push([len, right/len]);
    console.log(len, right);
    console.log(linelist);
  });
});
function drawCurveTypes() {

      var data = new google.visualization.DataTable();

      data.addColumn('number', 'X');
      data.addColumn('number', 'sketchmind');
      
      data.addRows(linelist);
      console.log(linelist);


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

      var chart = new google.visualization.LineChart(document.getElementById('linechart_div'));
      chart.draw(data, options);
    }