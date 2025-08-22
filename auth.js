/* Firebase Auth (Google + Email with verification) */
// Firebase 콘솔 → 프로젝트 설정 → 웹앱 추가 후 아래 구성값으로 교체
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  appId: "YOUR_APP_ID",
};
firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();

const byId = (id)=>document.getElementById(id);
const msg = (t)=> byId('authMsg').textContent = t || '';

byId('btnGoogle').onclick = async ()=>{
  try{
    const provider = new firebase.auth.GoogleAuthProvider();
    await auth.signInWithPopup(provider);
    msg('Google 로그인 성공. 메인으로 이동합니다.');
    location.href = './';
  }catch(e){ msg(e.message); }
};

byId('btnSignup').onclick = async ()=>{
  const email = byId('email').value.trim();
  const pw = byId('pw').value;
  if(pw.length < 8){ msg('비밀번호는 8자 이상'); return; }
  try{
    await auth.createUserWithEmailAndPassword(email, pw);
    await auth.currentUser.sendEmailVerification();
    msg('가입 완료. 이메일 인증 링크를 확인하세요.');
  }catch(e){ msg(e.message); }
};

byId('btnSignin').onclick = async ()=>{
  const email = byId('email').value.trim();
  const pw = byId('pw').value;
  try{
    const cred = await auth.signInWithEmailAndPassword(email, pw);
    if(!cred.user.emailVerified){
      await auth.signOut();
      msg('이메일 미인증. 받은 메일의 링크로 인증하세요.');
      return;
    }
    msg('로그인 성공. 메인으로 이동합니다.');
    location.href = './';
  }catch(e){ msg(e.message); }
};

byId('btnResend').onclick = async ()=>{
  try{
    if(auth.currentUser) { await auth.currentUser.sendEmailVerification(); msg('재전송 완료'); }
    else { msg('먼저 로그인하거나 가입하세요.'); }
  }catch(e){ msg(e.message); }
};
