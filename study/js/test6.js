// alert("ReRad1");
// alert("ReRad2");
// alert("ReRad3");

// var data = confirm("ReRad OK?");
// if (data == true){
//     alert("ReRad OK!");
//     console.log(data);
// }

// var date = prompt("Скажите сколько вам лет? ");
// console.log(date);

var person = null;
if (confirm("Вы точно уверены? ")){
    person = prompt("Введите ваше имя");
    alert("Привет, " + person);
}   else {
alert("Вы нажали на Отмена");
}
