// var count = 0;

// function onClickButton(){
//    count++;
//    console.log(count)
// }

var count = 0;

function onClickButton(el){
   count++;
   el.innerHTML = "Вы нажали на кнопку: " + count;
   el.style.background = "red";
   el.style.color = "blue";

   el.style.cssText = "border-radius: 5px; border: 0; font-size: 20px; background: green";
   console.log(el.name);
   console.log(el.onclick);
   console.log(el.value);
}

function onInput(el) {
    if (el.value == "Hello")
        alert("Hello!!!");
    console.log(el.value);
}


