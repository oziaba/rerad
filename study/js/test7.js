// function info(){
//     console.log("Привет");
//     console.log("!");
// }

// info();
// info();

// function info(word){
//     console.log(word + "!");
// }

// // info("ReRad");

// function summa(a,b){
//     var res = a + b;
//     console.log(res);
//     info(res);
// }

// summa(5,7);


// function summa(arr){
//     var sum = 0;

//     for(var i = 0; i < arr.length; i++){
//         sum += arr[i];
//     }
//     console.log(sum);
// }

var array_1 = [6,2,7];
var array_2 = [6,2,7,2];
var array_3 = [6,2,7,2,7];

// summa(array_1)
// summa(array_2)
// summa(array_3)


function summa(arr){
    var sum = 0;

    for(var i = 0; i < arr.length; i++){
        sum += arr[i];
    }
    return sum;
}

var res_1 = summa(array_1);
console.log(res_1);
var res_2 = summa(array_2);
var res_3 = summa(array_3);
console.log(res_2 + res_3);
