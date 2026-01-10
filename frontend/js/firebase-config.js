// Firebase Configuration
const firebaseConfig = {
    apiKey: "AIzaSyAQir-bun8eQU-nq22o8HIAOEzJ4SWNNQE",
    authDomain: "resume-screening-17cff.firebaseapp.com",
    projectId: "resume-screening-17cff",
    storageBucket: "resume-screening-17cff.firebasestorage.app",
    messagingSenderId: "321615453928",
    appId: "1:321615453928:web:b785a7cc3ea7c21758b35d",
    measurementId: "G-NJTDEK3CYB"
};

// Initialize Firebase
// We use the compat library (namespaced) to work easily with the script tags in HTML
if (typeof firebase !== 'undefined') {
    if (!firebase.apps.length) {
        firebase.initializeApp(firebaseConfig);
    }
    // Make auth and db available globally or just use firebase.auth() directly
    // console.log("Firebase Initialized with Project:", firebaseConfig.projectId);
} else {
    console.error("Firebase SDK not loaded. Make sure script tags are included in the HTML.");
}
