<?php
$servername = "localhost:3306";
$username = "nidheesh_nataraj";
$password = "qwerty";
$database="nidheesh_nataraj";

$conn = mysqli_connect($servername, $username, $password,$database);
echo "success";
$firstname = $_POST['firtname'];
$lastname = $_POST['lastname'];
$email= $_POST['email'];
$username = $_POST['username'];
$Gender=$_POST['gender'];
$dob=$_POST['dob'];
$dob = strtotime($dob);
$dob= date('d-m-y', $dob);
$password=$_POST['password'];
 

$sql = "Insert into signuptable(firstname,lastname,email,username,gender,dob, password) values ('$firstname','$lastname','$email', '$username', '$Gender','$dob','$password')";

if ($conn->query($sql) === TRUE) {
    echo  "sign up successful";
  
    
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

$conn->close();
?>

