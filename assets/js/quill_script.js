$("#text2json").click(function () { text2json() });

function text2json() {
    var jsonString = JSON.stringify(quill.getContents())
    var jsonContainer = document.getElementById("json-container");
    jsonContainer.textContent = jsonString;
};

$("#json2text").click(function () { json2text() });
function json2text() {
    var jsonContainer = document.getElementById("json-container");
    console.log(jsonContainer.value)
    quill.setContents(JSON.parse(jsonContainer.value));
};