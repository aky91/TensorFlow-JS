
function setup() {

    noCanvas();
    
    const values = [];
    
    for(let i = 0; i < 15; i++)
        values[i] = random(0, 100);
    
    const shape = [5, 3];
    
//    const tense = tf.tensor3d(values, shape, 'int32');
    
//    const vtense = tf.variable(tense);
//    console.log(vtense);
    
//    tense.print();
//    console.log(tens.data());
//    console.log(tens.toString());
//    console.log(tens);
    
/*    tense.data().then(function(stuff){
        console.log(stuff);
    });*/
    
//    console.log(tense.dataSync());
//    tense.print();
//    console.log(tense.get());
    
    const a = tf.tensor2d(values, shape, 'int32');
    const b = tf.tensor2d(values, shape, 'int32');
    
    const bb = b.transpose();
    
//    const c = a.mul(b);
    
    const c = a.matMul(bb);

    
    a.print();
    b.print();
    c.print();


    
}
