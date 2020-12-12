
var list = [];
$.ajaxSetup({ cache: false });
$.getJSON('../static/history.json', function(data) {         
    $.each( data.history, function( key, val ){
        list.push( val );
        console.log(val);
    });
    
});
var query = window.location.search.substring(1);
var Field=query.split("=");
var prediction = Field[1].split('&')[0];
var real = Field[2];


    // Start file download.
window.onload=function(){
    if (real === prediction) {
        document.getElementById("result_text").innerHTML = "You are correct!";
        console.log(list[list.length-1].predicted)
        list[list.length-1].predicted = prediction
        var obj = new Object();
        obj.history = list
        var obj_s = JSON.stringify(obj);
    } 
    else {
        document.getElementById("result_text").innerHTML = "You are wrong!";
        document.getElementById("result_text2").innerHTML = "It is a(an) " + real + "!";
        console.log(list[list.length-1].predicted)
        list[list.length-1].predicted = prediction
        list[list.length-1].correctness = "wrong!";
        var obj = new Object();
        obj.history = list
        var obj_s = JSON.stringify(obj);
    }
    function download(filename, text) {
        var element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
        element.setAttribute('download', filename);

        element.style.display = 'none';
        document.body.appendChild(element);

        element.click();

        document.body.removeChild(element);
    }
    download("history.json",obj_s);                    
    console.log(Field);
}

