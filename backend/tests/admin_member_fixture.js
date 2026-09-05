/* Synthetic, in-memory data only. Shared by the DOM harness and loopback preview. */
window.installAdminFixture = function(options = {}) {
  const uid = '10000000-0000-4000-8000-000000000001';
  const other = '20000000-0000-4000-8000-000000000002';
  const requests = [], holds = new Map(), failures = new Map();
  const users = [
    {user_id: uid, email: 'mei.lin@example.invalid', display_name: 'Mei Lin', practice_count: 100, is_pro_effective: true, is_pro_paid: true, is_pro_grant: true, pro_grant_at: '2026-08-20T06:00:00Z', pro_grant_by: 'operator@example.invalid', pro_grant_reason: '學習計畫延長', pro_grant_expires_at: '2099-10-05T00:00:00Z', created_at: '2026-04-21T00:00:00Z', last_practice: '2026-09-05T08:10:00Z'},
    {user_id: other, email: 'alex.chen@example.invalid', display_name: 'Alex Chen', practice_count: 0, is_pro_effective: false, is_pro_paid: false, is_pro_grant: false, created_at: '2026-09-02T00:00:00Z'},
    ...['jamie.wu', 'sophie.huang', 'ryan.liu', 'olivia.chang', 'ethan.wang', 'isabel.chen'].map((name, i) => ({user_id: `30000000-0000-4000-8000-00000000000${i}`, email: `${name}@example.invalid`, practice_count: 28-i*3, is_pro_effective: i%2===0, last_practice: '2026-09-04T00:00:00Z', created_at: '2026-06-01T00:00:00Z'})),
  ];
  const speaking = Array.from({length:100}, (_, i) => ({id: `record-${i}`, user_id: uid, mode: i%3===0?'part2':'part1', question: ['Describe a place where you enjoy spending time.', 'What do you enjoy most about your daily routine?', 'How has your hometown changed in recent years?'][i%3], user_transcript: 'I enjoy spending time near the river because it gives me a chance to slow down and notice the city around me.', coach_response: 'Use a specific example to develop this idea.', weakness_tag: ['lack_detail','safe_answer','weak_vocab'][i%3], quality_grade: ['valid','partial','invalid'][i%3], created_at: new Date(Date.UTC(2026,8,5,8)-i*86400000).toISOString()}));
  const writing = Array.from({length:4}, (_, i) => ({id:`essay-${i}`, task_type: i%2?'task2':'task1', submitted_at: new Date(Date.UTC(2026,8,4)-i*86400000).toISOString(), writing_questions: {prompt: i%2?'Some people believe that cities should invest more in public transport. Discuss both views.':'The chart shows how household spending changed over time.', task1_subtype: i%2?null:'bar_chart'}, band_overall: 6+i/2, band_ta:6, essay_text:'The chart illustrates changes in household spending. Overall, housing remained the largest category.', feedback_ta:'Make the overview clearer.', priority_fix:'Compare the largest changes.', word_count: 182+i*40}));
  const reading = Array.from({length:23}, (_, i) => ({id:`reading-${i}`, started_at:new Date(Date.UTC(2026,8,4)-i*86400000).toISOString(), submitted_at:'2026-09-04T12:00:00Z', reading_passages:{title:['The changing shape of urban life','How forests communicate','The history of public libraries'][i%3], difficulty_band:'B2', word_count:650}, status:i===2?'in_progress':'submitted', score:i===2?null:8, total:10, answer_count:i===2?0:10, band_estimate:6.5}));
  const subscriptions = [{id:'sub-1', user_id:uid, order_id:'LOCAL-ORDER-001', plan:'monthly', status:'active', amount:299, started_at:'2026-08-05T00:00:00Z', expires_at:'2099-10-05T00:00:00Z'}];
  window.BLABBY_CONFIG = {supabaseUrl:'http://127.0.0.1', supabaseAnonKey:'synthetic'};
  window.BLABBY_API_BASE = 'http://127.0.0.1:8765/api-fixture';
  window.supabase = {createClient: () => ({auth: {getSession: async () => ({
    data: {session: options.signedOut ? null : {user: {email: 'authorized-by-server@example.invalid'}, access_token: 'synthetic'}}
  })}})};
  window.confirm = () => true;
  window.alert = message => { window.fixture.alerts.push(message); };
  window.fetch = async (url, init = {}) => {
    const parsed = new URL(url); const path = parsed.pathname.replace('/api-fixture','');
    requests.push({path, query:parsed.search, method:init.method || 'GET', body:init.body ? JSON.parse(init.body):null, headers:init.headers});
    for (const [match, hold] of holds) if (path.includes(match)) await hold.promise;
    for (const [match, status] of failures) if (path.includes(match)) return {ok:false, status, json:async()=>({detail:`Fixture failure ${status}`})};
    if (options.denied) return {ok:false,status:403,json:async()=>({detail:'Admin access required'})};
    let body;
    if (path === '/admin/users') body={users};
    else if (/\/pro_grant$/.test(path)) {const u=users.find(u=>path.includes(u.user_id));const data=JSON.parse(init.body);Object.assign(u,{is_pro_grant:data.granted, is_pro_effective:u.is_pro_paid || data.granted, pro_grant_expires_at:data.expires_at});body={status:'ok'};}
    else if (path.includes('/student_brief/')) body={brief:'現況判斷：已有穩定練習。建議加入具體細節。'};
    else if (path.endsWith('/diagnosis')) body={format:'structured',generated_at:'2026-09-05T10:00:00Z',data:{summary:'口說表達清楚，細節仍可加强。',weaknesses:[{rank:1,title:'缺少具體細節',evidence:['描述家鄉時缺乏例子。']}],next_step:'用一個親身經驗延伸回答。'}};
    else if (path === '/admin/writing/submissions') body={submissions:parsed.searchParams.get('user_id')===other?[]:writing};
    else if (path === '/admin/reading/attempts') body={attempts:parsed.searchParams.get('user_id')===other?[]:reading};
    else if (path === '/api/admin/subscriptions') body={subscriptions:subscriptions.filter(s=>s.user_id===parsed.searchParams.get('user_id'))};
    else if (path === '/api/admin/subscription/extend' || path === '/api/admin/subscription/cancel') body={status:'ok'};
    else if (path === '/admin/practice-volume') body={days:[]};
    else if (path.startsWith('/admin/user/')) {const offset=Number(parsed.searchParams.get('offset')||0);const records=path.includes(uid)?speaking:[];body={user_id:path.split('/').pop(),records:records.slice(offset,offset+10),total_records:records.length,offset,has_more:offset+10<records.length,weakness_counts:records.length?{lack_detail:37,off_topic:21,safe_answer:15,weak_vocab:13,grammar_minor:3}:{}};}
    else throw new Error(`Unexpected fixture request: ${path}`);
    return {ok:true,status:200,json:async()=>JSON.parse(JSON.stringify(body))};
  };
  window.fixture = {uid,other,requests,failures,users,alerts:[],hold(match){let release;const promise=new Promise(resolve=>{release=resolve;});holds.set(match,{promise});return ()=>{holds.delete(match);release();};}};
};
