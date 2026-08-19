/**
 * フォーム自動保存スクリプト。
 *
 * `data-autosave` 属性を持つ <form> を対象に、フィールドの変更を検知して
 * フォーム全体を action へ POST する（Save ボタンを押さずに即時反映する）。
 *
 * - テキスト入力は入力停止後 debounce、選択・チェック・ファイルは即時保存する。
 * - 既存のフォーム送信ハンドラをそのまま再利用する（X-Requested-With ヘッダで
 *   AJAX 判定し、サーバは JSON を返す）。
 * - `data-no-autosave` を持つ要素の変更は無視する（クロップ前の生画像など）。
 * - 画像クロップ確定など、イベントを伴わない変更は window.chotgorAutosave() で
 *   明示的に保存をトリガーできる。
 *
 * 楽観ロック（先祖返り防止）:
 *   フォーム全体を送る性質上、放置したタブが他端末の変更を丸ごと巻き戻しうる。
 *   サーバは hidden `_fp`（対象フィールドの現在値の指紋）を照合し、ずれていれば
 *   409 を返す。ここではその 409 を受けて自動保存を止め、バナーで
 *   「このまま上書き」（`_fp_force`）か「破棄して読み直す」を選ばせる。
 *   保存成功時はレスポンスの新しい指紋を hidden へ書き戻し、自分自身の連続保存が
 *   衝突しないようにする。サーバ側の対は backend/lib/optimistic_lock.py。
 */
(function () {
  "use strict";

  /** テキスト入力の保存を遅延させる時間（ミリ秒）。 */
  var DEBOUNCE_MS = 700;

  /** 楽観ロックの hidden フィールド名（backend/lib/optimistic_lock.py と対）。 */
  var FP_FIELD = "_fp";
  var FORCE_FIELD = "_fp_force";

  /** サーバがメッセージを返さなかったときの衝突文言。 */
  var DEFAULT_CONFLICT_MESSAGE =
    "この設定は別の端末（または別のタブ）で変更されている。" +
    "このまま保存すると、そちらの変更が巻き戻る。";

  /** 画面右下に出す保存ステータス表示要素（遅延生成）。 */
  var statusEl = null;

  /** 画面下部に出す衝突バナー（同時に 1 つだけ）。 */
  var conflictEl = null;

  /** 保存ステータス表示要素を取得する（無ければ生成する）。 */
  function ensureStatus() {
    if (statusEl) return statusEl;
    statusEl = document.createElement("div");
    statusEl.className = "autosave-status";
    document.body.appendChild(statusEl);
    return statusEl;
  }

  /**
   * 保存ステータスを表示する。
   * @param {string} text 表示文言
   * @param {string} kind saving | saved | error | invalid
   */
  function showStatus(text, kind) {
    var el = ensureStatus();
    el.textContent = text;
    el.dataset.kind = kind;
    el.classList.add("visible");
    clearTimeout(el._hideTimer);
    // 完了系の表示は一定時間で自動的に消す。
    if (kind === "saved" || kind === "invalid" || kind === "error") {
      el._hideTimer = setTimeout(function () {
        el.classList.remove("visible");
      }, 2200);
    }
  }

  /** 衝突バナーを閉じる。 */
  function closeConflict() {
    if (conflictEl && conflictEl.parentNode) {
      conflictEl.parentNode.removeChild(conflictEl);
    }
    conflictEl = null;
  }

  /**
   * 衝突バナーを出す。ユーザが選ぶまで自動保存は止まったままになる。
   * @param {string} message サーバから返された文言
   * @param {() => void} onOverwrite 「このまま上書きする」を選んだときの処理
   */
  function openConflict(message, onOverwrite) {
    closeConflict();
    var el = document.createElement("div");
    el.className = "ch-notice ch-notice--warn ch-notice--float";

    var text = document.createElement("div");
    text.textContent = message || DEFAULT_CONFLICT_MESSAGE;

    var actions = document.createElement("div");
    actions.className = "ch-row-gap";
    actions.style.marginTop = "10px";

    var overwrite = document.createElement("button");
    overwrite.type = "button";
    overwrite.className = "ch-btn ch-btn--danger ch-btn--sm";
    overwrite.textContent = "このまま上書きする";
    overwrite.addEventListener("click", function () {
      closeConflict();
      onOverwrite();
    });

    var reload = document.createElement("button");
    reload.type = "button";
    reload.className = "ch-btn ch-btn--ghost ch-btn--sm";
    reload.textContent = "破棄して読み直す";
    reload.addEventListener("click", function () {
      location.reload();
    });

    actions.appendChild(overwrite);
    actions.appendChild(reload);
    el.appendChild(text);
    el.appendChild(actions);
    document.body.appendChild(el);
    conflictEl = el;
  }

  /** 保存レスポンスで返ってきた新しい指紋をフォームの hidden へ書き戻す。 */
  function applyFingerprint(form, fp) {
    if (!fp) return;
    var el = form.querySelector('input[name="' + FP_FIELD + '"]');
    if (el) el.value = fp;
  }

  /**
   * 1 つのフォームに自動保存の挙動を取り付ける。
   * @param {HTMLFormElement} form
   */
  function attach(form) {
    var timer = null;
    var saving = false;
    var pending = false;
    // 衝突を検出したら、ユーザが選ぶまで自動保存を止める（勝手に上書きしない）。
    var blocked = false;

    /**
     * フォーム内容を POST する。保存中の多重実行は pending に畳む。
     * @param {boolean} [force] true なら衝突を承知で上書きする
     * @returns {Promise<boolean>} 保存できたら true
     */
    function save(force) {
      if (blocked && !force) return Promise.resolve(false);
      if (saving) {
        pending = true;
        return Promise.resolve(false);
      }
      // 不正な入力なら保存しない（ブラウザの検証 + setCustomValidity を流用）。
      // 最初の不正フィールドの validationMessage を出す（無ければ既定文言）。
      if (typeof form.checkValidity === "function" && !form.checkValidity()) {
        var invalid = form.querySelector(":invalid");
        var msg = (invalid && invalid.validationMessage) || "必須項目が未入力です";
        showStatus(msg, "invalid");
        return Promise.resolve(false);
      }
      saving = true;
      showStatus("保存中…", "saving");
      var fd = new FormData(form);
      if (force) fd.append(FORCE_FIELD, "1");
      var conflicted = false;
      var saved = false;
      return fetch(form.action, {
        method: "POST",
        body: fd,
        headers: { "X-Requested-With": "fetch" },
      })
        .then(function (res) {
          if (res.status === 409) {
            conflicted = true;
          } else if (!res.ok) {
            throw new Error("HTTP " + res.status);
          }
          return res.json().catch(function () {
            return {};
          });
        })
        .then(function (data) {
          if (conflicted) {
            blocked = true;
            pending = false;
            showStatus("他の端末の変更を検出。保存を止めた", "error");
            openConflict(data && data.message, function () {
              save(true);
            });
            return;
          }
          blocked = false;
          closeConflict();
          applyFingerprint(form, data && data.fp);
          saved = true;
          showStatus("保存しました", "saved");
        })
        .catch(function () {
          showStatus("保存に失敗しました", "error");
        })
        .then(function () {
          saving = false;
          if (pending && !blocked) {
            pending = false;
            save();
          }
          return saved;
        });
    }

    // 外部スクリプト（画像まわりの独自送信など）から強制保存を呼べるようにする。
    form._chotgorSave = save;

    /**
     * 保存をスケジュールする。
     * @param {boolean} immediate true なら debounce せず即時保存する
     */
    function schedule(immediate) {
      clearTimeout(timer);
      if (immediate) {
        save();
      } else {
        timer = setTimeout(save, DEBOUNCE_MS);
      }
    }

    // テキスト入力中は debounce 保存する。
    form.addEventListener("input", function (e) {
      if (e.target.closest("[data-no-autosave]")) return;
      schedule(false);
    });

    // 選択・チェック・ファイルは確定操作なので即時保存する。
    form.addEventListener("change", function (e) {
      if (e.target.closest("[data-no-autosave]")) return;
      var t = e.target;
      var immediate =
        t.tagName === "SELECT" ||
        t.type === "checkbox" ||
        t.type === "radio" ||
        t.type === "file";
      schedule(immediate);
    });

    // 画像クロップ確定など、イベントを伴わない変更からの明示トリガー。
    form.addEventListener("autosave:now", function () {
      schedule(true);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var forms = document.querySelectorAll("form[data-autosave]");
    for (var i = 0; i < forms.length; i++) attach(forms[i]);
  });

  /** フォーム要素かセレクタから <form> を解決する。 */
  function resolveForm(formOrSelector) {
    return typeof formOrSelector === "string"
      ? document.querySelector(formOrSelector)
      : formOrSelector;
  }

  /**
   * 外部スクリプトから保存を明示的にトリガーする。
   * @param {HTMLFormElement|string} formOrSelector フォーム要素か CSS セレクタ
   */
  window.chotgorAutosave = function (formOrSelector) {
    var form = resolveForm(formOrSelector);
    if (form) form.dispatchEvent(new CustomEvent("autosave:now"));
  };

  /**
   * 独自の送信処理（画像アップロードなど）が 409 を受け取ったときに、
   * 自動保存と同じ衝突バナーを出す。
   * @param {HTMLFormElement|string} formOrSelector 対象フォーム
   * @param {string} [message] サーバから返された文言
   * @param {() => void} [afterOverwrite] 上書き保存が成功した後の処理
   */
  window.chotgorConflict = function (formOrSelector, message, afterOverwrite) {
    var form = resolveForm(formOrSelector);
    if (!form) return;
    openConflict(message, function () {
      var result =
        typeof form._chotgorSave === "function" ? form._chotgorSave(true) : null;
      if (result && typeof result.then === "function") {
        result.then(function (ok) {
          if (ok && afterOverwrite) afterOverwrite();
        });
      }
    });
  };
})();
