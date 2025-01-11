use backprop::{network};

fn main (){
   // 3 layer network.
    let network = network::Network{
        layers: vec![
            network::Layer::new(3, 4),
            network::Layer::new(4, 5),
            network::Layer::new(5, 100),
            network::Layer::new(100, 1),
        ],
    };
    
    let inputs = vec![0.1, 0.2, 0.3];
    
    let _ = network.forward(&inputs);

    println!("Hi");
}