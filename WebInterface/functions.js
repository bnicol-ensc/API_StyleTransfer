function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#imageInput").attr("src", e.target.result);
    };

    reader.readAsDataURL(input.files[0]);
  }
}

function callPostAPI() {
  const formData = new FormData();
  const fileField = document.querySelector('input[type="file"]');

  formData.append("file", fileField.files[0]);

  var style = $("#chosenStyle").val();
  console.log(style);

  fetch("http://localhost:5000/model/" + style, {
    method: "POST",
    body: formData,
  })
    .then((data) => data.blob())
    .then((images) => {
      // Then create a local URL for that image and print it
      outside = URL.createObjectURL(images);
      document.getElementById("imageOutput").src = outside;
    })
    .catch((error) => {
      console.error(error);
    });
}

$("select").imagepicker();

console.log($("#chosenStyle").val());
