function change_URLs() {
    var link_element = document.getElementById("switch_lang");
    var current_URL = window.location.href;
    if(current_URL.includes("/en/")) {
        link_element.href = current_URL.replace("/en/", "/fr/").replace("/en/", "/fr/");
    } else {
        link_element.href = current_URL.replace("/fr/", "/en/").replace("/fr/", "/en/");
    }
}
  

window.onload = function() {
    change_URLs();
}