var num = 15;

if (num != 15) {
    console.log("OK");
} else {
    console.log("Else!");
}

if (num != 15) {
    console.log("OK");
} else if(num < 10) {
    console.log("< 10")
} else if(num == 7) {
    console.log("= 7");
} else if(num > 15) {
    console.log("> 15");
} else {
    console.log("Error!");
}


var num_1 = 5;
var flag = true;

if (num_1 == 5 && !flag) {
    console.log("OK1");
}
if (num_1 == 5 && flag) {
    console.log("OK2");
}

var stroka = "word1"

switch(stroka){
    case "4":
        console.log("Переменная со значением 4");
        break;
    case "42":
        console.log("Переменная со значением 42");
        break;
    case "word":
        console.log("Переменная со значением word");
        break;
    default:
        console.log("Default");
        break;
}
