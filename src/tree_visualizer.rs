use crate::decision_tree::DecisionTree;

pub fn print_tree(tree: &DecisionTree, depth: usize) {
    if let Some(label) = tree.label {
        println!("{}Leaf: Class {}", " ".repeat(depth * 4), label);
        return;
    }
    println!(
        "{}Feature {}: <= {}",
        " ".repeat(depth * 4),
        tree.feature,
        tree.threshold
    );
    println!("{}Left:", " ".repeat(depth * 4));
    print_tree(tree.left.as_ref().unwrap(), depth + 1);
    println!("{}Right:", " ".repeat(depth * 4));
    print_tree(tree.right.as_ref().unwrap(), depth + 1);
}
