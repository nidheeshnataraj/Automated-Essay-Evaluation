<!DOCTYPE html>
<html>
<head>
  <script src="http://code.jquery.com/jquery-1.9.1.js"></script>
</head>
<body>
    <form id="mForm" action="signup.php" title="" method="post">
        <div>
            <label class="font">Firstname</label>
            <input type="text" id="FIRST NAME" name="firstname" >
        </div>
        <div>
            <label class="font">Lastname</label>
            <input type="text" id="LAST NAME" name="lastname" >
        </div>
         <div>
            <label class="font">Email Address</label>
            <input type="text" id="EMAIL ADDRESS" name="email" >
        </div>
		 <div>
            <label class="font">Username</label>
            <input type="text" id="username" name="username" >
        </div>
		 <div>
            <label class="font">Birthday</label>
            <input type="text" id="DOB" name="dob" >
        </div>
		 <div>
            <label class="font">Password:"</label>
            <input type="text" id="password" name="password" >
        </div>
		<div>
        <input type="Submit" id="Submit"  name="Submit" value="Submit">
    </div>
 </form>
<script type='text/javascript'>
    /* attach a submit handler to the form */
    $("#formoid").submit(function(event) {

      /* stop form from submitting normally */
      event.preventDefault();

      /* get the action attribute from the <form action=""> element */
      var $form = $( this ),
          url = $form.attr( 'action' );

      /* Send the data using post with element id name and name2*/
      var posting = $.post( url, { name: $('#name').val(), name2: $('#name2').val() } );

      /* Alerts the results */
      posting.done(function( data ) {
        alert('success');
      });
    });
</script>

</body>
</html> 