// let counter = 0; 
// var id = setInterval(my_func, 1000);

// function my_func(){
//     counter++;
//     console.log("Counter: " + counter);

//     if (counter == 3)
//         clearInterval(id);
    
// }
// setInterval(function(){
//     counter++;
//     console.log("Counter: " + counter);
// }, 1500);


setTimeout(function(){
    console.log("Timer is working!");
}, 2000);

setTimeout(my_func, 2000);
function my_func(){
    console.log("Timer is working! my_func")
}